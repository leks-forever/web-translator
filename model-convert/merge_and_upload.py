"""
Merges decoder_model.onnx + decoder_with_past_model.onnx into
decoder_model_merged.onnx, quantizes it to int8, and uploads to HF.
"""
import os
import sys
import glob

MODEL_ID = "leks-forever/nllb-200-distilled-600M"
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
    merged_path = os.path.join(ONNX_DIR, "decoder_model_merged.onnx")
    merged_data_path = os.path.join(ONNX_DIR, "decoder_model_merged.onnx_data")
    if os.path.exists(merged_path):
        log("==> decoder_model_merged.onnx already exists, skipping merge")
    else:
        log("==> Merging decoder_model.onnx + decoder_with_past_model.onnx ...")
        import onnx
        from optimum.onnx.graph_transformations import merge_decoders

        # Use save_path=None to get the ModelProto back without saving.
        # This avoids the ir_version validation error and the 2GB protobuf limit.
        merged = merge_decoders(
            decoder=os.path.join(ONNX_DIR, "decoder_model.onnx"),
            decoder_with_past=os.path.join(ONNX_DIR, "decoder_with_past_model.onnx"),
            save_path=None,
            strict=False,
        )

        # Fix ir_version (merge_decoders leaves it unset → checker rejects it)
        merged.ir_version = 8
        log(f"  Merged graph: {len(merged.graph.node)} nodes")

        # Save with external data to avoid the 2GB protobuf inline limit
        log("==> Saving merged model with external data ...")
        onnx.save_model(
            merged,
            merged_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="decoder_model_merged.onnx_data",
            convert_attribute=False,
        )
        graph_mb = os.path.getsize(merged_path) / 1e6
        data_mb  = os.path.getsize(merged_data_path) / 1e6
        log(f"  Saved: decoder_model_merged.onnx ({graph_mb:.0f} MB graph + {data_mb:.0f} MB data)")

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
        size_mb = os.path.getsize(path) / 1e6
        log(f"  Uploading onnx/{fname} ({size_mb:.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"onnx/{fname}",
            repo_id=MODEL_ID,
            repo_type="model",
        )

    log("\n==> Done!")
    log(f"    https://huggingface.co/leks-forever/nllb-200-distilled-600M/tree/main/onnx")


if __name__ == "__main__":
    main()
