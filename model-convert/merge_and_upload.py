"""
Merges decoder_model.onnx + decoder_with_past_model.onnx into
decoder_model_merged.onnx, quantizes it to int8, and uploads to HF.
"""
import os
import sys
import glob

MODEL_ID = "leks-forever/nllb-200-distilled-600M-v1"
ONNX_DIR = "./onnx_export/onnx"


def log(msg):
    print(msg, flush=True)


def download_if_needed():
    """Download decoder ONNX files from HF if not already present locally."""
    from huggingface_hub import hf_hub_download
    needed = [
        "onnx/decoder_model.onnx",
        "onnx/decoder_model.onnx_data",
        "onnx/decoder_with_past_model.onnx",
        "onnx/decoder_with_past_model.onnx_data",
    ]
    os.makedirs(ONNX_DIR, exist_ok=True)
    for repo_path in needed:
        local = os.path.join(ONNX_DIR, os.path.basename(repo_path))
        if os.path.exists(local):
            log(f"  {os.path.basename(local)} already present, skipping download")
            continue
        log(f"  Downloading {repo_path} ...")
        tmp = hf_hub_download(
            repo_id=MODEL_ID,
            filename=repo_path,
            repo_type="model",
        )
        import shutil
        shutil.copy(tmp, local)
        log(f"  Saved to {local}")


def main():
    from huggingface_hub import login
    login(token=os.environ.get("HF_TOKEN"))

    log("==> Checking / downloading source ONNX files ...")
    download_if_needed()

    # Step 1: Merge decoder + decoder_with_past into merged decoder
    # Low-memory strategy: load only graph structure (no weights), patch
    # numpy_helper.to_array so deduplication never reads from disk, then
    # symlink the existing decoder_model.onnx_data as the merged data file.
    # Peak RAM: ~200 MB instead of ~9 GB.
    merged_path = os.path.join(ONNX_DIR, "decoder_model_merged.onnx")
    merged_data_path = os.path.join(ONNX_DIR, "decoder_model_merged.onnx_data")
    if os.path.exists(merged_path):
        log("==> decoder_model_merged.onnx already exists, skipping merge")
    else:
        log("==> Merging (low-memory, graph-only) ...")
        import onnx
        import numpy as np
        import onnx.numpy_helper as nh
        from onnx import TensorProto

        # Patch to_array: return a hash-based sentinel for external tensors
        # so same-named tensors compare equal without loading 3 GB from disk.
        _orig_to_array = nh.to_array
        def _patched_to_array(tensor, base_dir=""):
            if getattr(tensor, "data_location", None) == TensorProto.EXTERNAL:
                return np.array([hash(tensor.name) & 0x7FFFFFFF], dtype=np.int64)
            return _orig_to_array(tensor, base_dir)
        nh.to_array = _patched_to_array

        # Must cd into ONNX dir so relative external_data paths resolve
        orig_cwd = os.getcwd()
        os.chdir(ONNX_DIR)

        from optimum.onnx.graph_transformations import merge_decoders
        decoder    = onnx.load("decoder_model.onnx",    load_external_data=False)
        decoder_wp = onnx.load("decoder_with_past_model.onnx", load_external_data=False)
        log(f"  decoder: {len(decoder.graph.node)} nodes / "
            f"decoder_with_past: {len(decoder_wp.graph.node)} nodes")

        merged = merge_decoders(decoder=decoder, decoder_with_past=decoder_wp,
                                save_path=None, strict=False)
        nh.to_array = _orig_to_array  # restore patch
        os.chdir(orig_cwd)

        merged.ir_version = 8
        log(f"  Merged: {len(merged.graph.node)} nodes, "
            f"{len(merged.graph.initializer)} initializers")

        # Repoint all external_data locations to decoder_model_merged.onnx_data
        for t in merged.graph.initializer:
            if t.data_location == TensorProto.EXTERNAL:
                for e in t.external_data:
                    if e.key == "location":
                        e.value = "decoder_model_merged.onnx_data"

        # Save only the tiny graph file (no weight data copied)
        log("==> Saving merged graph file ...")
        onnx.save_model(merged, merged_path, save_as_external_data=False)
        graph_kb = os.path.getsize(merged_path) / 1e3
        log(f"  Saved: decoder_model_merged.onnx ({graph_kb:.0f} kB)")

        # Symlink decoder_model.onnx_data → decoder_model_merged.onnx_data
        # (all shared weights stay in place, no 3 GB copy needed)
        if not os.path.exists(merged_data_path):
            os.symlink(os.path.abspath(os.path.join(ONNX_DIR, "decoder_model.onnx_data")),
                       merged_data_path)
            log(f"  Symlinked decoder_model.onnx_data → decoder_model_merged.onnx_data")

    # Step 2: Quantize merged decoder
    merged_q_path = os.path.join(ONNX_DIR, "decoder_model_merged_quantized.onnx")
    if os.path.exists(merged_q_path):
        log(f"==> decoder_model_merged_quantized.onnx already exists, skipping quantization")
    else:
        log("==> Quantizing decoder_model_merged.onnx ...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(merged_path, merged_q_path, weight_type=QuantType.QUInt8)
        size_mb = os.path.getsize(merged_q_path) / 1e6
        log(f"  Saved: decoder_model_merged_quantized.onnx ({size_mb:.0f} MB)")

    # Step 3: Upload both files (and any associated .onnx_data)
    log(f"\n==> Uploading to {MODEL_ID}/onnx/ ...")
    from huggingface_hub import HfApi
    api = HfApi()

    to_upload = []
    for pattern in [f"{ONNX_DIR}/decoder_model_merged*.onnx", f"{ONNX_DIR}/decoder_model_merged*.onnx_data"]:
        to_upload.extend(glob.glob(pattern))

    for path in sorted(to_upload):
        fname = os.path.basename(path)
        real  = os.path.realpath(path)  # follow symlinks for .onnx_data
        size_mb = os.path.getsize(real) / 1e6
        log(f"  Uploading onnx/{fname} ({size_mb:.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=real,
            path_in_repo=f"onnx/{fname}",
            repo_id=MODEL_ID,
            repo_type="model",
        )

    log("\n==> Done!")
    log(f"    https://huggingface.co/leks-forever/nllb-200-distilled-600M/tree/main/onnx")


if __name__ == "__main__":
    main()
