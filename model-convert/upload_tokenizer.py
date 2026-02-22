"""
Generates tokenizer.json (fast tokenizer) from the SentencePiece model
and uploads all tokenizer files to the HF repo root.
"""
import os
from huggingface_hub import login, HfApi

MODEL_ID = "leks-forever/nllb-200-distilled-600M-v1"
OUT_DIR = "./tokenizer_out"


def main():
    token = os.environ.get("HF_TOKEN")
    login(token=token)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("==> Loading fast tokenizer...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tok.save_pretrained(OUT_DIR)

    print("\n==> Saved files:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f}  ({size / 1e3:.1f} kB)")

    print(f"\n==> Uploading tokenizer files to {MODEL_ID} root...")
    api = HfApi()
    for fname in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, fname)
        print(f"  Uploading {fname} ...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=fname,
            repo_id=MODEL_ID,
            repo_type="model",
        )

    print("\n==> Done!")


if __name__ == "__main__":
    main()
