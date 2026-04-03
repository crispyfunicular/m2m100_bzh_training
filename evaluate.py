#!/usr/bin/env python3
"""
evaluate.py — Évaluation du modèle unifié M2M100 (br→fr)

Métriques : chrF2 + BLEU (sacrebleu) sur Flores-200 devtest
            (bre_Latn→fra_Latn, 1 012 paires)

Prérequis :
  pip install sacrebleu transformers torch datasets

Usage :
  python3 evaluate.py
  python3 evaluate.py --batch-size 16
"""

import os
import argparse
import pathlib
import sys

# Force le cache HuggingFace vers le répertoire personnel de l'utilisateur
# pour éviter les erreurs de permission sur les caches partagés de laboratoire.
_user_hf_cache = str(pathlib.Path.home() / ".cache" / "huggingface")
os.environ["HF_HOME"] = _user_hf_cache
os.environ["TRANSFORMERS_CACHE"] = _user_hf_cache

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# ── Constantes ────────────────────────────────────────────────────────────────

SCRIPT_DIR = pathlib.Path(__file__).parent

BASELINE_MODEL   = "facebook/m2m100_418M"
UNIFIED_MODEL_DIR = SCRIPT_DIR / "output" / "unified"

# zeldarose sauvegarde dans un sous-répertoire "muppet"
OUTPUT_SUBDIR = "muppet"

MODELS = [
    {"label": "Baseline (facebook/m2m100_418M)",       "path": BASELINE_MODEL},
    {"label": "Unifié (Kenstur + OCR phrases)",         "path": None},  # résolu ci-dessous
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_unified_model() -> str | None:
    """Retourne le chemin du modèle unifié fine-tuné (output/unified/)."""
    for candidate in [UNIFIED_MODEL_DIR / OUTPUT_SUBDIR, UNIFIED_MODEL_DIR]:
        if (candidate / "config.json").exists():
            return str(candidate)
    return None


def load_flores_data() -> list[dict]:
    """Charge Flores-200 devtest (bre_Latn→fra_Latn) depuis HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ⚠️  Package 'datasets' manquant. Installez-le avec : pip install datasets",
              file=sys.stderr)
        sys.exit(1)
    print("  Chargement Flores-200 (bre_Latn / fra_Latn, devtest)...")
    br = load_dataset("facebook/flores", "bre_Latn", split="devtest", trust_remote_code=True)
    fr = load_dataset("facebook/flores", "fra_Latn", split="devtest", trust_remote_code=True)
    pairs = [{"br": b["sentence"], "fr": f["sentence"]} for b, f in zip(br, fr)]
    print(f"  {len(pairs)} paires Flores-200 chargées.\n")
    return pairs


def generate_translations(
    model_path: str,
    sources: list[str],
    batch_size: int,
    device: str,
) -> list[str]:
    """Génère les traductions br→fr pour une liste de phrases source."""
    print(f"  Chargement du modèle : {model_path}")
    tokenizer = M2M100Tokenizer.from_pretrained(model_path)
    model = M2M100ForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    tokenizer.src_lang = "br"
    lang_id = tokenizer.get_lang_id("fr")

    hypotheses = []
    total = len(sources)
    for i in range(0, total, batch_size):
        batch = sources[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=lang_id,
                num_beams=5,
                max_new_tokens=256,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        hypotheses.extend(decoded)
        done = min(i + batch_size, total)
        print(f"    {done}/{total} phrases traduites", end="\r")
    print()

    # Libère la mémoire GPU entre les modèles
    del model
    torch.cuda.empty_cache()

    return hypotheses


# ── Métriques ─────────────────────────────────────────────────────────────────

def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu as sb
    return sb.corpus_bleu(hypotheses, [references]).score


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu as sb
    return sb.corpus_chrf(hypotheses, [references], beta=2).score


# ── Affichage ─────────────────────────────────────────────────────────────────

def print_results_table(results: list[dict]) -> None:
    header = f"\n{'─'*60}\n  Résultats\n{'─'*60}"
    print(header)

    cols = ["Modèle", "chrF2", "BLEU"]
    col_w = [45, 8, 8]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
    sep = "  " + "  ".join("─" * w for w in col_w)

    print(fmt.format(*cols))
    print(sep)
    for r in results:
        print(fmt.format(r["label"][:col_w[0]], f"{r['chrf2']:.2f}", f"{r['bleu']:.2f}"))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évalue le modèle M2M100 unifié contre la baseline (br→fr, Flores-200)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        dest="batch_size",
        help="Taille de batch pour la génération (défaut : 8).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Dispositif : {device.upper()}")
    print(f"  Évaluation : Flores-200 devtest (bre_Latn→fra_Latn, 1 012 paires)\n")

    # Résolution du modèle unifié
    unified_path = resolve_unified_model()
    MODELS[1]["path"] = unified_path

    # Chargement Flores-200 (une seule fois)
    pairs = load_flores_data()
    sources    = [p["br"] for p in pairs]
    references = [p["fr"] for p in pairs]

    results = []

    for model_info in MODELS:
        label = model_info["label"]
        path  = model_info["path"]

        print(f"{'═'*60}")
        print(f"  {label}")
        print(f"{'═'*60}")

        if path is None:
            print(f"Modèle unifié introuvable dans {UNIFIED_MODEL_DIR} — ignoré.\n")
            continue

        hypotheses = generate_translations(path, sources, args.batch_size, device)

        chrf2 = compute_chrf(hypotheses, references)
        bleu  = compute_bleu(hypotheses, references)
        print(f"  chrF2 : {chrf2:.2f}  |  BLEU : {bleu:.2f}\n")

        results.append({"label": label, "chrf2": chrf2, "bleu": bleu})

    if results:
        print_results_table(results)
    else:
        print("Aucun résultat à afficher.")


if __name__ == "__main__":
    main()
