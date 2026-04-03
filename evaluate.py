#!/usr/bin/env python3
"""
evaluate.py — Évaluation d'un modèle M2M100 fine-tuné (br→fr)

Métriques calculées sur deux jeux d'évaluation :
  • chrF2 + BLEU sur dev.jsonl (corpus vu pendant l'entraînement)
  • chrF2 + BLEU sur Flores-200 devtest (bre_Latn→fra_Latn, 1 012 paires)

Prérequis :
  pip install sacrebleu transformers torch datasets

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

# Force le cache HuggingFace vers le répertoire personnel de l'utilisateur
# pour éviter les erreurs de permission sur les caches partagés de laboratoire.
_user_hf_cache = str(pathlib.Path.home() / ".cache" / "huggingface")
os.environ["HF_HOME"] = _user_hf_cache
os.environ["TRANSFORMERS_CACHE"] = _user_hf_cache

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# zeldarose sauvegarde dans un sous-répertoire "muppet"
OUTPUT_SUBDIR = "muppet"


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_model(output_dir: pathlib.Path) -> str | None:
    """Retourne le chemin du modèle fine-tuné (muppet/ ou racine du dossier)."""
    for candidate in [output_dir / OUTPUT_SUBDIR, output_dir]:
        if (candidate / "config.json").exists():
            return str(candidate)
    return None


def load_local_dev(output_dir: pathlib.Path) -> list[dict]:
    """Charge dev.jsonl depuis le dossier output."""
    path = output_dir / "dev.jsonl"
    if not path.exists():
        print(f"  ⚠️  dev.jsonl introuvable dans {output_dir}", file=sys.stderr)
        return []
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Supporte les deux formats : {"br":..., "fr":...} et {"translation":{...}}
            if "translation" in d:
                d = d["translation"]
            pairs.append({"br": d["br"].strip(), "fr": d["fr"].strip()})
    return pairs


def load_flores_data() -> list[dict]:
    """Charge Flores-200 devtest (bre_Latn→fra_Latn) depuis HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ⚠️  Package 'datasets' manquant : pip install datasets", file=sys.stderr)
        sys.exit(1)
    print("  Chargement Flores-200 (bre_Latn / fra_Latn, devtest)...")
    br = load_dataset("facebook/flores", "bre_Latn", split="devtest", trust_remote_code=True)
    fr = load_dataset("facebook/flores", "fra_Latn", split="devtest", trust_remote_code=True)
    pairs = [{"br": b["sentence"], "fr": f["sentence"]} for b, f in zip(br, fr)]
    print(f"  {len(pairs)} paires chargées.\n")
    return pairs


def generate_translations(
    model_path: str,
    sources: list[str],
    batch_size: int,
    device: str,
) -> list[str]:
    """Génère les traductions br→fr pour une liste de phrases source."""
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

def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu as sb
    return round(sb.corpus_chrf(hypotheses, [references], beta=2).score, 2)


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu as sb
    return round(sb.corpus_bleu(hypotheses, [references]).score, 2)


def evaluate_on(
    model_path: str,
    pairs: list[dict],
    batch_size: int,
    device: str,
    label: str,
) -> dict:
    """Génère et calcule chrF2 + BLEU sur un jeu de paires."""
    print(f"\n── {label} ({len(pairs)} paires) ─────────────────────")
    sources    = [p["br"] for p in pairs]
    references = [p["fr"] for p in pairs]
    hypotheses = generate_translations(model_path, sources, batch_size, device)
    chrf2 = compute_chrf(hypotheses, references)
    bleu  = compute_bleu(hypotheses, references)
    print(f"  chrF2 : {chrf2:.2f}  |  BLEU : {bleu:.2f}")
    return {"eval_set": label, "n_pairs": len(pairs), "chrf2": chrf2, "bleu": bleu}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évalue un modèle M2M100 fine-tuné (br→fr) : dev.jsonl + Flores-200."
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="Dossier de sortie de l'entraînement (contient le modèle, dev.jsonl, etc.).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        dest="batch_size",
        help="Taille de batch pour la génération (défaut : 8).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        print(f"Erreur : dossier introuvable — {output_dir}", file=sys.stderr)
        sys.exit(1)

    model_path = resolve_model(output_dir)
    if model_path is None:
        print(f"Erreur : aucun modèle trouvé dans {output_dir}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'═'*60}")
    print(f"  Modèle    : {model_path}")
    print(f"  Dispositif: {device.upper()}")
    print(f"{'═'*60}")

    # ── Chargement des données ─────────────────────────────────────────────────
    dev_pairs    = load_local_dev(output_dir)
    flores_pairs = load_flores_data()

    # ── Évaluation ─────────────────────────────────────────────────────────────
    results = []

    if dev_pairs:
        results.append(evaluate_on(
            model_path, dev_pairs, args.batch_size, device,
            label="dev.jsonl"
        ))

    results.append(evaluate_on(
        model_path, flores_pairs, args.batch_size, device,
        label="Flores-200 devtest"
    ))

    # ── Tableau récapitulatif ──────────────────────────────────────────────────
    print(f"\n{'─'*60}\n  Résultats — {output_dir.name}\n{'─'*60}")
    col_w = [22, 8, 8, 8]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format("Jeu d'évaluation", "Paires", "chrF2", "BLEU"))
    print("  " + "  ".join("─" * w for w in col_w))
    for r in results:
        print(fmt.format(r["eval_set"][:col_w[0]], str(r["n_pairs"]),
                         f"{r['chrf2']:.2f}", f"{r['bleu']:.2f}"))
    print()

    # ── Sauvegarde evaluation.json ─────────────────────────────────────────────
    report = {
        "model_dir": str(output_dir),
        "model_name": output_dir.name,
        "language":   "br→fr",
        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        "results":    results,
    }
    eval_path = output_dir / "evaluation.json"
    eval_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Résultats sauvegardés dans : {eval_path}\n")


if __name__ == "__main__":
    main()
