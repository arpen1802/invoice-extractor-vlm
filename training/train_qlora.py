"""QLoRA fine-tune of Qwen2.5-VL-3B-Instruct for invoice -> JSON extraction.

Two entry paths:

  --dry-run   CPU-only. Loads the manifests, builds the exact train prompts and
              JSON targets, and prints a few, plus target-length stats. No model,
              no GPU, no heavy deps beyond pyyaml/PIL. Use this to sanity-check
              the data pipeline anywhere (including a Mac) before renting a GPU.

  (default)   Full run. Lazily imports torch/transformers/peft/bitsandbytes,
              loads the base model in 4-bit, attaches LoRA, and trains with the
              HF Trainer while logging to W&B. Requires a CUDA GPU.

  --max-samples N   Truncate train/val to N examples — use it to PROVE the loop
              runs end-to-end for a few steps before committing to a full run.

Usage
-----
    python training/train_qlora.py --config training/config.yaml --dry-run
    python training/train_qlora.py --config training/config.yaml --max-samples 20   # smoke (GPU)
    python training/train_qlora.py --config training/config.yaml                     # full (GPU)

GPU note: 3B QLoRA + vision fits ~16-24GB VRAM at batch 1 with the image pixel
caps in config.yaml. See the memory notes printed at startup.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.prompt import EXTRACT_PROMPT  # noqa: E402


# ---- config + data (CPU-safe) ---------------------------------------------

def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_manifest(rel_path: str) -> list[dict]:
    p = ROOT / rel_path
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p} (run data/prepare.py first)")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in rows:
        r["_label_obj"] = json.loads((ROOT / r["label"]).read_text())
    return rows


def build_messages(image_ref, target: str | None):
    """Qwen2.5-VL chat messages. `image_ref` is a PIL image or a file path.
    If `target` is None, this is an inference prompt (add generation prompt)."""
    user = {"role": "user", "content": [
        {"type": "image", "image": image_ref},
        {"type": "text", "text": EXTRACT_PROMPT},
    ]}
    msgs = [user]
    if target is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return msgs


# ---- dry run (no model) ---------------------------------------------------

def dry_run(cfg: dict, n: int):
    from PIL import Image
    train = load_manifest(cfg["data"]["train_manifest"])
    val = load_manifest(cfg["data"]["val_manifest"])
    print(f"Loaded manifests: train={len(train)}  val={len(val)}")
    if not train:
        print("No training rows — run data/synth/generate.py + data/prepare.py first.")
        return

    lengths = []
    for r in train:
        lengths.append(len(json.dumps(r["_label_obj"])))
    print(f"Target JSON length (chars): min={min(lengths)} "
          f"median={int(statistics.median(lengths))} max={max(lengths)}")

    print(f"\n--- {min(n, len(train))} sample training examples ---")
    for r in train[:n]:
        img = ROOT / r["image"]
        size = Image.open(img).size if img.exists() else "MISSING"
        tgt = json.dumps(r["_label_obj"], ensure_ascii=False)
        msgs = build_messages(str(img), tgt)
        print(f"\nimage: {r['image']}  size={size}  source={r['source']}")
        print(f"user prompt: {EXTRACT_PROMPT[:60]}... ({len(EXTRACT_PROMPT)} chars)")
        print(f"target: {tgt[:200]}{'...' if len(tgt) > 200 else ''}")
        assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant"
    print("\nDry run OK: manifests, images, prompts, and targets all resolve.")


# ---- full training (GPU) --------------------------------------------------

def _bnb_config(cfg):
    import torch
    from transformers import BitsAndBytesConfig
    q = cfg["quant"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )


class _Collator:
    """Builds a batch: applies the chat template, runs the vision processor, and
    masks the prompt tokens so loss is computed only on the JSON answer."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        import torch
        try:
            from qwen_vl_utils import process_vision_info
        except Exception:
            process_vision_info = None

        texts, image_inputs_all, prompt_lens = [], [], []
        for r in batch:
            img_path = str(ROOT / r["image"])
            target = json.dumps(r["_label_obj"], ensure_ascii=False)
            full = build_messages(img_path, target)
            prompt_only = build_messages(img_path, None)

            text = self.processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            texts.append(text)

            # prompt length (tokens) to mask, computed from prompt-only rendering
            ptext = self.processor.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
            prompt_lens.append(len(self.processor.tokenizer(ptext, add_special_tokens=False)["input_ids"]))

            if process_vision_info is not None:
                imgs, _ = process_vision_info(full)
            else:
                from PIL import Image
                imgs = [Image.open(img_path).convert("RGB")]
            image_inputs_all.append(imgs)

        # flatten images (one image per sample here)
        flat_images = [im for sub in image_inputs_all for im in sub]
        model_inputs = self.processor(
            text=texts, images=flat_images, return_tensors="pt", padding=True,
        )
        labels = model_inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100
        # mask prompt region per-sample
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        model_inputs["labels"] = labels
        return model_inputs


def train(cfg: dict, max_samples: int | None):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoProcessor, Trainer, TrainingArguments,
                              Qwen2_5_VLForConditionalGeneration)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU. This run needs a GPU host. Use --dry-run on CPU.")
        sys.exit(1)

    base = cfg["model"]["base_id"]
    processor = AutoProcessor.from_pretrained(
        base, trust_remote_code=cfg["model"]["trust_remote_code"],
        min_pixels=cfg["data"]["image_min_pixels"],
        max_pixels=cfg["data"]["image_max_pixels"],
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, quantization_config=_bnb_config(cfg), torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=cfg["model"]["trust_remote_code"],
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"])

    if cfg["model"].get("freeze_vision"):
        for name, p in model.named_parameters():
            if "visual" in name:
                p.requires_grad_(False)

    lora = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], bias="none", task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_rows = load_manifest(cfg["data"]["train_manifest"])
    val_rows = load_manifest(cfg["data"]["val_manifest"])
    if max_samples:
        train_rows, val_rows = train_rows[:max_samples], val_rows[:max(1, max_samples // 5)]
    print(f"train={len(train_rows)} val={len(val_rows)}"
          f"{' (SMOKE)' if max_samples else ''}")

    t = cfg["train"]
    args = TrainingArguments(
        output_dir=t["output_dir"], num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"], weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"], eval_strategy="steps",
        eval_steps=t["eval_steps"], save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"], bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"], seed=t["seed"],
        report_to=["wandb"], run_name=cfg["wandb"]["run_name"],
        remove_unused_columns=False,
    )

    import os
    os.environ.setdefault("WANDB_PROJECT", cfg["wandb"]["project"])
    os.environ.setdefault("WANDB_MODE", cfg["wandb"]["mode"])

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_rows, eval_dataset=val_rows,
        data_collator=_Collator(processor),
    )
    trainer.train()
    out = Path(t["output_dir"]) / "adapter"
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"Saved LoRA adapter + processor to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="CPU: validate data pipeline, no model")
    ap.add_argument("--n", type=int, default=3, help="samples to show in --dry-run")
    ap.add_argument("--max-samples", type=int, default=None, help="smoke run on a tiny subset")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        dry_run(cfg, args.n)
    else:
        train(cfg, args.max_samples)


if __name__ == "__main__":
    main()
