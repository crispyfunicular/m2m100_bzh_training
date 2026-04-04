#!/usr/bin/env python3
"""
evaluate.py — Évaluation d'un modèle M2M100 fine-tuné (br→fr)

Métriques : chrF2 + BLEU sur dev.jsonl

Prérequis :
  pip install sacrebleu transformers torch

Usage :
  python3 evaluate.py output/1_korpusou
  python3 evaluate.py output/1_korpusou --batch-size 16
"""

import argparse
import datetime
import json
import os
import pathlib
import sys

_user_hf_cache = str(pathlib.Path.home() / ".cache" / "huggingface")
os.environ["HF_HOME"] = _user_hf_cache
os.environ["TRANSFORMERS_CACHE"] = _user_hf_cache

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

OUTPUT_SUBDIR = "muppet"


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_model(output_dir: pathlib.Path) -> str | None:
    for candidate in [output_dir / OUTPUT_SUBDIR, output_dir]:
        if (candidate / "config.json").exists():
            return str(candidate)
    return None


def load_jsonl(path: pathlib.Path) -> list[dict]:
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "translation" in d:
                d = d["translation"]
            br = d.get("br", "").strip()
            fr = d.get("fr", "").strip()
            if br and fr:
                pairs.append({"br": br, "fr": fr})
    return pairs


def generate_translations(model_path, sources, batch_size, device):
    print(f"  Chargement : {model_path}")
    tokenizer = M2M100Tokenizer.from_pretrained(model_path)
    model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    tokenizer.src_lang = "br"
    lang_id = tokenizer.get_lang_id("fr")
    hypotheses = []
    total = len(sources)
    for i in range(0, total, batch_size):
        batch = sources[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, forced_bos_token_id=lang_id, num_beams=5, max_new_tokens=256
            )
        hypotheses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        print(f"    {min(i + batch_size, total)}/{total} phrases traduites", end="\r")
    print()
    del model
    torch.cuda.empty_cache()
    return hypotheses


# ── Métriques ─────────────────────────────────────────────────────────────────

def compute_chrf(hypotheses, references):
    import sacrebleu as sb
    return round(sb.corpus_chrf(hypotheses, [references], beta=2).score, 2)


def compute_bleu(hypotheses, references):
    import sacrebleu as sb
    return round(sb.corpus_bleu(hypotheses, [references]).score, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Évalue un modèle M2M100 fine-tuné (br→fr) — chrF2 + BLEU sur dev.jsonl."
    )
    parser.add_argument("output_dir", type=pathlib.Path,
                        help="Dossier de sortie de l'entraînement.")
    parser.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    parser.add_argument("--eval-file", default="dev.jsonl", dest="eval_file",
                        help="Fichier à évaluer (défaut : dev.jsonl).")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        print(f"Erreur : dossier introuvable — {output_dir}", file=sys.stderr)
        sys.exit(1)

    model_path = resolve_model(output_dir)
    if model_path is None:
        print(f"Erreur : aucun modèle trouvé dans {output_dir}", file=sys.stderr)
        sys.exit(1)

    dev_path = output_dir / args.eval_file
    if not dev_path.exists():
        print(f"Erreur : {args.eval_file} introuvable dans {output_dir}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Modèle    : {model_path}")
    print(f"  Dispositif: {device.upper()}\n")

    pairs      = load_jsonl(dev_path)
    sources    = [p["br"] for p in pairs]
    references = [p["fr"] for p in pairs]

    hypotheses = generate_translations(model_path, sources, args.batch_size, device)
    chrf2 = compute_chrf(hypotheses, references)
    bleu  = compute_bleu(hypotheses, references)

    print(f"\n  chrF2 : {chrf2:.2f}")
    print(f"  BLEU  : {bleu:.2f}\n")

    report = {
        "model_dir":  str(output_dir),
        "model_name": output_dir.name,
        "language":   "br→fr",
        "eval_set":   args.eval_file,
        "n_pairs":    len(pairs),
        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        "chrf2":      chrf2,
        "bleu":       bleu,
    }
    eval_path = output_dir / "evaluation.json"
    eval_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Résultats sauvegardés dans : {eval_path}\n")


if __name__ == "__main__":
    main()
