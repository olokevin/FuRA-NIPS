"""Merge a QLoRA adapter into the base model, save as a full HF model.

Reads:  --adapter_dir (a PEFT save dir with adapter_config.json + adapter_model.safetensors)
Writes: --output_dir as a full HF model (config.json, pytorch_model.bin or shards)
"""
import argparse
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    print(f"Loading base model from {args.base_model} ...")
    # Note: load in bf16 to match training and avoid quantized state on disk.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading tokenizer from {args.adapter_dir} (saved by PEFT) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

    # Match what finetune_qlora.py did: resize_token_embeddings to a multiple of 8.
    target_vocab = int(8 * math.ceil(len(tokenizer) / 8.0))
    if model.get_input_embeddings().num_embeddings != target_vocab:
        print(f"Resizing embeddings: {model.get_input_embeddings().num_embeddings} -> {target_vocab}")
        model.resize_token_embeddings(target_vocab)

    print(f"Loading PEFT adapter from {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)

    print("Merging adapter into base ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
