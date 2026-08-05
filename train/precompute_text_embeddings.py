import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm


def load_prompt_schedule(dataset_json: str,
                         splits: List[str] | None = None
                         ) -> List[Tuple[str, str, str]]:
    """Expand a ReXGroundingCT dataset JSON to a flat work-list.

    Args:
        dataset_json: path to a JSON whose top-level keys are split names
            (e.g. {"train": [...], "val": [...]}), each split entry being a
            dict with keys "name" and "findings" where "findings" is itself
            a dict {finding_idx_str: prompt_text}.
        splits: optional explicit list of split keys to process.  If None
            every top-level key present in the JSON is used.

    Returns:
        flat list of 3-tuples (image_name_stem, prompt_text, finding_idx_str)
        ready for the encoder loop.
    """
    with open(dataset_json, "r", encoding="utf-8") as f:
        dataset_dict = json.load(f)

    if splits is None:
        splits = list(dataset_dict.keys())
    else:
        for s in splits:
            if s not in dataset_dict:
                raise KeyError(
                    f"--splits asked for '{s}' but available splits in "
                    f"'{dataset_json}' are: {list(dataset_dict.keys())}"
                )

    prompts: List[Tuple[str, str, str]] = []
    for split_name in splits:
        entries = dataset_dict[split_name]
        for data in tqdm(entries, desc=f"collecting {split_name}"):
            name = str(data["name"]).replace(".nii.gz", "")
            findings: Dict[str, str] = data.get("findings", {}) or {}
            for idx_str, prompt_text in findings.items():
                prompts.append((name, str(prompt_text), str(idx_str)))
    return prompts


def encode_and_save(prompts: List[Tuple[str, str, str]],
                    model_name_or_path: str,
                    output_dir: str,
                    gpu_id: int,
                    *,
                    max_seq_len: int = 256,
                    skip_existing: bool = True,
                    batch_size: int = 1) -> None:
    """Encode every prompt in `prompts` and write individual .pt files.

    Args:
        prompts: flat work-list from `load_prompt_schedule`.
        model_name_or_path: HF repo / local folder for Qwen3 AutoModel / AutoTokenizer.
        output_dir: destination directory; created if missing.
        gpu_id: CUDA ordinal (0-based).
        max_seq_len: tokenizer truncation / padding length.
        skip_existing: if True, any `{name}_{idx}.pt` that already exists
            with the right keys is skipped (safe resume on interrupt).
        batch_size: how many prompts to batch into a single encoder forward.
            Use 1 for safety on 4B models; bump if you have enough VRAM.
    """
    # --- Local import: keep the import off cold paths so `--help` is fast. ---
    from model.prompt_seg.text_encoder import Qwen3TextEncoder

    os.makedirs(output_dir, exist_ok=True)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu"

    text_encoder = Qwen3TextEncoder(
        model_name_or_path=model_name_or_path,
        freeze=True,
        max_seq_len=max_seq_len,
    )
    text_encoder.eval()
    text_encoder.to(device)

    # --- Optionally prune the work-list if resume requested ---------------
    if skip_existing:
        remaining: List[Tuple[str, str, str]] = []
        skipped = 0
        for name, prompt_text, idx in prompts:
            target = os.path.join(output_dir, f"{name}_{idx}.pt")
            if os.path.exists(target):
                try:
                    probe = torch.load(target, map_location="cpu")
                    if {"text_emb", "text_tokens", "text_mask"} <= set(probe.keys()):
                        skipped += 1
                        continue
                except Exception:  
                    pass
            remaining.append((name, prompt_text, idx))
        if skipped:
            print(f"[precompute_text_embeddings] skip {skipped} existing files; "
                  f"{len(remaining)} prompts still need encoding.")
        prompts = remaining

    # --- Encode in batches ------------------------------------------------
    total = len(prompts)
    if total == 0:
        print("[precompute_text_embeddings] nothing to do — all prompts encoded.")
        return

    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc="encoding"):
            batch = prompts[start:start + batch_size]
            names = [b[0] for b in batch]
            texts = [b[1] for b in batch]
            idxs = [b[2] for b in batch]

            text_emb, text_tokens, text_mask = text_encoder(texts, device=device)

            for i in range(len(batch)):
                save_path = os.path.join(output_dir, f"{names[i]}_{idxs[i]}.pt")
                payload = {
                    "text_emb":   text_emb[i].detach().cpu(),
                    "text_tokens": text_tokens[i].detach().cpu(),
                    "text_mask":   text_mask[i].detach().cpu(),
                }
                tmp_path = save_path + ".tmp"
                torch.save(payload, tmp_path)
                os.replace(tmp_path, save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute and persist Qwen3 prompt embeddings for "
                    "ReXGroundingCT training / inference.",
    )
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="Path to ReXGroundingCT-style dataset JSON (top-level split "
             "keys, each entry has 'name' and dict 'findings').",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HF repo id or local folder for the Qwen3 AutoModel/AutoTokenizer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where {name}_{idx}.pt files are written.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="CUDA device ordinal (default 0; -1 → CPU).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Optional whitelist of dataset JSON top-level splits to encode "
             "(space-separated).  Default = encode every split present.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256,
        help="Tokenizer truncation length (default 256).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Prompts per encoder forward pass (default 1; safe on 4B models; "
             "increase for smaller models / larger GPUs).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode and overwrite already-existing .pt files "
             "(default behaviour is to skip them for resume).",
    )

    args = parser.parse_args()

    prompts = load_prompt_schedule(args.dataset_json, splits=args.splits)
    print(f"[precompute_text_embeddings] collected {len(prompts)} prompt(s) "
          f"from '{args.dataset_json}'.")

    encode_and_save(
        prompts=prompts,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        max_seq_len=args.max_seq_len,
        skip_existing=(not args.overwrite),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
