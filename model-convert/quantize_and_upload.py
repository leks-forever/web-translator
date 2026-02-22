"""
Quantizes the already-exported ONNX files to int8 and uploads
everything (fp32 + quantized) to HuggingFace.

Run after convert.py has exported the ONNX files:
    HF_TOKEN=hf_... uv run quantize_and_upload.py
"""

import glob
import os
import sys

MODEL_ID = "leks-forever/nllb-200-distilled-600M-v1"
ONNX_DIR = "./onnx_export/onnx"

# Files to quantize (skip the -inferred helper file)
MODELS_TO_QUANTIZE = [
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
]


def main() -> None:
    print("==> Logging in to HuggingFace...")
    from huggingface_hub import login
    token = os.environ.get("HF_TOKEN")
    login(token=token)

    # Step 1: Quantize each model to int8
    print("\n==> Quantizing ONNX models to int8 (_quantized.onnx)...")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    for fname in MODELS_TO_QUANTIZE:
        src = os.path.join(ONNX_DIR, fname)
        stem = fname.replace(".onnx", "")
        dst = os.path.join(ONNX_DIR, f"{stem}_quantized.onnx")

        if os.path.exists(dst):
            print(f"  Skipping {fname} (quantized file already exists)")
            continue

        print(f"  {fname} -> {os.path.basename(dst)}")
        quantize_dynamic(src, dst, weight_type=QuantType.QUInt8)

    print("\n==> Quantization done. Files:")
    for f in sorted(glob.glob(f"{ONNX_DIR}/*.onnx")):
        size = os.path.getsize(f)
        print(f"  {os.path.basename(f):50s} {size / 1e6:.1f} MB")

    # Step 2: Upload everything to HuggingFace onnx/ subfolder
    # Upload order: quantized first (smaller), then fp32
    print(f"\n==> Uploading to {MODEL_ID}/onnx/ ...")
    from huggingface_hub import HfApi
    api = HfApi()

    # Collect files to upload: all .onnx and .onnx_data files except the -inferred helper
    upload_files = []
    for f in sorted(glob.glob(f"{ONNX_DIR}/*.onnx") + glob.glob(f"{ONNX_DIR}/*.onnx_data")):
        if "inferred" in os.path.basename(f):
            continue
        upload_files.append(f)

    for path in upload_files:
        fname = os.path.basename(path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  Uploading onnx/{fname} ({size_mb:.0f} MB) ...")
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
