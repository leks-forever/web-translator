"""
Memory-efficient decoder merge.
Loads only the ONNX graph structure (not external weight data),
merges the graphs, then stitches in the original external data references.
Peak memory: ~200MB instead of ~6GB.
"""
import os
import sys
import glob
import shutil

MODEL_ID = "leks-forever/nllb-200-distilled-600M-v1"
ONNX_DIR = "./onnx_export/onnx"


def log(msg):
    print(msg, flush=True)


def build_ext_map(model):
    """Return {tensor_name: external_data_entries} from an onnx ModelProto
    loaded with load_external_data=False."""
    from onnx import TensorProto
    mapping = {}
    for t in model.graph.initializer:
        if t.data_location == TensorProto.EXTERNAL:
            mapping[t.name] = list(t.external_data)
    return mapping


def restore_ext_refs(merged_model, ext_map_a, ext_map_b):
    """Put external-data references back on each initializer in merged model."""
    from onnx import TensorProto
    missing = []
    for t in merged_model.graph.initializer:
        if t.name in ext_map_a:
            t.data_location = TensorProto.EXTERNAL
            del t.external_data[:]
            t.external_data.extend(ext_map_a[t.name])
        elif t.name in ext_map_b:
            t.data_location = TensorProto.EXTERNAL
            del t.external_data[:]
            t.external_data.extend(ext_map_b[t.name])
        else:
            missing.append(t.name)
    if missing:
        log(f"  WARNING: {len(missing)} tensors have no external data ref (likely small inlined tensors, OK)")
    return merged_model


def main():
    import onnx
    from optimum.onnx.graph_transformations import merge_decoders
    from huggingface_hub import login
    login(token=os.environ.get("HF_TOKEN"))

    dec_path = os.path.join(ONNX_DIR, "decoder_model.onnx")
    dec_wp_path = os.path.join(ONNX_DIR, "decoder_with_past_model.onnx")
    merged_path = os.path.join(ONNX_DIR, "decoder_model_merged.onnx")
    merged_q_path = os.path.join(ONNX_DIR, "decoder_model_merged_quantized.onnx")

    # ---------- Step 1: merge graphs without loading weights ----------
    if os.path.exists(merged_path):
        log("==> decoder_model_merged.onnx already exists, skipping merge")
    else:
        log("==> Loading decoder graphs (no external data) ...")
        decoder    = onnx.load(dec_path,    load_external_data=False)
        decoder_wp = onnx.load(dec_wp_path, load_external_data=False)
        log(f"  Loaded: {len(decoder.graph.node)} nodes / {len(decoder_wp.graph.node)} nodes")

        ext_a = build_ext_map(decoder)
        ext_b = build_ext_map(decoder_wp)
        log(f"  External tensors: {len(ext_a)} in decoder, {len(ext_b)} in decoder_with_past")

        log("==> Merging graph structures ...")
        merged = merge_decoders(
            decoder=decoder,
            decoder_with_past=decoder_wp,
            strict=False,
        )
        log(f"  Merged graph: {len(merged.graph.node)} nodes")

        log("==> Restoring external data references ...")
        merged = restore_ext_refs(merged, ext_a, ext_b)

        # The merged model references two separate .onnx_data files.
        # We need it to work as a single unit. Copy decoder_model.onnx_data
        # as decoder_model_merged.onnx_data, and patch references that
        # point to decoder_with_past_model.onnx_data.
        #
        # Since NLLB decoder and decoder_with_past share ALL weights,
        # we only need one .onnx_data file. Verify assumption:
        only_in_b = set(ext_b.keys()) - set(ext_a.keys())
        log(f"  Tensors only in decoder_with_past (not shared): {len(only_in_b)}")
        if only_in_b:
            log(f"  Unique tensors: {list(only_in_b)[:5]}")

        if only_in_b:
            log("  Cannot reuse single data file — need to combine external data.")
            log("  Creating combined data file (streaming copy) ...")
            merged_data_path = os.path.join(ONNX_DIR, "decoder_model_merged.onnx_data")

            # Copy decoder_model.onnx_data as the base
            dec_data = os.path.join(ONNX_DIR, "decoder_model.onnx_data")
            dec_wp_data = os.path.join(ONNX_DIR, "decoder_with_past_model.onnx_data")

            shutil.copy2(dec_data, merged_data_path)
            base_offset = os.path.getsize(merged_data_path)

            # Append unique tensors from decoder_with_past_model.onnx_data
            from onnx.external_data_helper import ExternalDataInfo
            offset_map = {}  # tensor_name -> new offset in merged file
            with open(merged_data_path, "ab") as out_f, open(dec_wp_data, "rb") as wp_f:
                for name in only_in_b:
                    entries = {e.key: e.value for e in ext_b[name]}
                    old_offset = int(entries.get("offset", 0))
                    length = int(entries.get("length", 0))
                    wp_f.seek(old_offset)
                    data = wp_f.read(length)
                    new_offset = base_offset + out_f.tell() - base_offset
                    out_f.seek(0, 2)  # seek to end
                    new_offset = out_f.tell()
                    out_f.write(data)
                    offset_map[name] = (new_offset, length)

            # Update references in merged model: all point to merged file
            for t in merged.graph.initializer:
                from onnx import TensorProto
                if t.data_location == TensorProto.EXTERNAL:
                    entries = {e.key: e.value for e in t.external_data}
                    loc = entries.get("location", "")
                    del t.external_data[:]
                    if t.name in offset_map:
                        new_off, new_len = offset_map[t.name]
                        t.external_data.add().CopyFrom(_kv("location", "decoder_model_merged.onnx_data"))
                        t.external_data.add().CopyFrom(_kv("offset", str(new_off)))
                        t.external_data.add().CopyFrom(_kv("length", str(new_len)))
                    else:
                        # Keep original offset in decoder_model.onnx_data, just rename
                        for k, v in entries.items():
                            if k == "location":
                                v = "decoder_model_merged.onnx_data"
                            t.external_data.add().CopyFrom(_kv(k, v))
        else:
            log("  All weights shared — symlinking decoder_model.onnx_data as merged data file")
            merged_data_path = os.path.join(ONNX_DIR, "decoder_model_merged.onnx_data")
            if not os.path.exists(merged_data_path):
                os.symlink(
                    os.path.abspath(os.path.join(ONNX_DIR, "decoder_model.onnx_data")),
                    merged_data_path,
                )
            # Patch all external_data references to point to merged data file name
            from onnx import TensorProto
            for t in merged.graph.initializer:
                if t.data_location == TensorProto.EXTERNAL:
                    for e in t.external_data:
                        if e.key == "location":
                            e.value = "decoder_model_merged.onnx_data"

        log(f"==> Saving decoder_model_merged.onnx ...")
        onnx.save_model(merged, merged_path, save_as_external_data=False)
        size_kb = os.path.getsize(merged_path) / 1e3
        log(f"  Saved: decoder_model_merged.onnx ({size_kb:.0f} kB graph + external data)")

    # ---------- Step 2: quantize ----------
    if os.path.exists(merged_q_path):
        log("==> decoder_model_merged_quantized.onnx already exists, skipping")
    else:
        log("==> Quantizing decoder_model_merged.onnx ...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(merged_path, merged_q_path, weight_type=QuantType.QUInt8)
        size_mb = os.path.getsize(merged_q_path) / 1e6
        log(f"  Saved: decoder_model_merged_quantized.onnx ({size_mb:.0f} MB)")

    # ---------- Step 3: upload ----------
    log(f"\n==> Uploading to {MODEL_ID}/onnx/ ...")
    from huggingface_hub import HfApi
    api = HfApi()
    to_upload = sorted(
        glob.glob(f"{ONNX_DIR}/decoder_model_merged*.onnx") +
        glob.glob(f"{ONNX_DIR}/decoder_model_merged*.onnx_data")
    )
    for path in to_upload:
        fname = os.path.basename(path)
        real_path = os.path.realpath(path)  # follow symlinks
        size_mb = os.path.getsize(real_path) / 1e6
        log(f"  Uploading onnx/{fname} ({size_mb:.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=real_path,
            path_in_repo=f"onnx/{fname}",
            repo_id=MODEL_ID,
            repo_type="model",
        )

    log("\n==> Done!")
    log(f"    https://huggingface.co/leks-forever/nllb-200-distilled-600M/tree/main/onnx")


def _kv(key, value):
    from onnx import StringStringEntryProto
    e = StringStringEntryProto()
    e.key = key
    e.value = value
    return e


if __name__ == "__main__":
    main()
