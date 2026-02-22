"""
Converts leks-forever/nllb-200-distilled-600M from safetensors to ONNX
and uploads the result back to the same HuggingFace repo.

Uses ORTModelForSeq2SeqLM (recommended Python API) which handles
the large model split correctly (encoder + decoder + decoder_with_past).
Then applies int8 dynamic quantization for browser/CPU use.

Usage:
    cd model-convert
    uv sync
    HF_TOKEN=hf_... uv run convert.py
"""

import glob
import os
import sys

MODEL_ID = "leks-forever/nllb-200-distilled-600M-v1"
EXPORT_DIR = "./onnx_export"


def main() -> None:
    # Step 1: Login
    print("==> Logging in to HuggingFace...")
    from huggingface_hub import login
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Tip: set HF_TOKEN env var to skip the interactive prompt.")
    login(token=token)

    # Step 2: Export to ONNX using the recommended Python API
    # ORTModelForSeq2SeqLM handles the seq2seq split automatically:
    # encoder_model.onnx + decoder_model.onnx + decoder_with_past_model.onnx
    print(f"\n==> Exporting {MODEL_ID} to ONNX (fp32)...")
    from optimum.onnxruntime import ORTModelForSeq2SeqLM

    ort_model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_ID, export=True)

    onnx_dir = f"{EXPORT_DIR}/onnx"
    os.makedirs(onnx_dir, exist_ok=True)
    # Save ONNX files immediately after export (before anything else can fail)
    ort_model.save_pretrained(onnx_dir)
    print(f"  Saved ONNX files to {onnx_dir}/")

    # Step 3: Quantize each ONNX file to int8 (_quantized.onnx)
    # transformers.js dtype='q8' loads files with _quantized suffix.
    print(f"\n==> Quantizing to int8 (dynamic quantization)...")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    source_files = sorted(glob.glob(f"{onnx_dir}/*.onnx"))
    for src in source_files:
        stem = os.path.basename(src).replace(".onnx", "")
        dst = os.path.join(onnx_dir, f"{stem}_quantized.onnx")
        print(f"  {os.path.basename(src)} -> {os.path.basename(dst)}")
        quantize_dynamic(src, dst, weight_type=QuantType.QUInt8)

    # Step 4: Upload all ONNX files to HuggingFace (inside onnx/ subfolder)
    print(f"\n==> Uploading ONNX files to {MODEL_ID}/onnx/ ...")
    from huggingface_hub import HfApi
    api = HfApi()

    all_onnx = sorted(glob.glob(f"{onnx_dir}/*.onnx"))
    for path in all_onnx:
        fname = os.path.basename(path)
        print(f"  Uploading onnx/{fname} ...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"onnx/{fname}",
            repo_id=MODEL_ID,
            repo_type="model",
        )

    print(f"\n==> Done!")
    print(f"    https://huggingface.co/{MODEL_ID}/tree/main/onnx")


if __name__ == "__main__":
    main()
