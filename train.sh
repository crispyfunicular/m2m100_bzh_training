#!/usr/bin/env bash
# train.sh — Fine-tuning M2M100 sur un corpus unique (br↔fr)
#
# Usage :
#   bash train.sh <dossier_corpus>
#
#   <dossier_corpus>   Dossier contenant config.toml, train.jsonl, dev.jsonl
#                      Ex : data/1_korpusou
#
# Sortie :
#   output/<nom_du_dossier>/   Modèle fine-tuné + copies de config.toml, train.jsonl, dev.jsonl
#
# Prérequis : pip install zeldarose sentencepiece
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Arguments ─────────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    echo "Usage : bash train.sh <dossier_corpus>"
    echo "  Ex  : bash train.sh data/1_korpusou"
    exit 1
fi

CORPUS_DIR="${1%/}"          # supprime le slash final si présent
PRETRAINED="facebook/m2m100_418M"
DATASET_NAME="$(basename "$CORPUS_DIR")"
OUT_DIR="output/$DATASET_NAME"

# ── Vérifications ─────────────────────────────────────────────────────────────

for required in config.toml train.jsonl dev.jsonl; do
    if [[ ! -f "$CORPUS_DIR/$required" ]]; then
        echo "Erreur : fichier manquant — $CORPUS_DIR/$required"
        exit 1
    fi
done

export HF_HOME="/home/mpellissier/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/mpellissier/.cache/huggingface/hub"
# Réduit la fragmentation mémoire GPU (recommandé par PyTorch pour les erreurs OOM)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Résumé ────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════"
echo "  Corpus   : $CORPUS_DIR"
echo "  Modèle   : $PRETRAINED"
echo "  Sortie   : $OUT_DIR"
echo "════════════════════════════════════════════════"
echo ""

# ── Entraînement ──────────────────────────────────────────────────────────────

zeldarose transformer \
    --tokenizer facebook/m2m100_418M \
    --pretrained-model "$PRETRAINED" \
    --config "$CORPUS_DIR/config.toml" \
    --out-dir "$OUT_DIR" \
    --val-text "$CORPUS_DIR/dev.jsonl" \
    "$CORPUS_DIR/train.jsonl"

# ── Archive dans le dossier de sortie ─────────────────────────────────────────
# Copie config + données pour reproductibilité

cp "$CORPUS_DIR/config.toml"  "$OUT_DIR/config.toml"
cp "$CORPUS_DIR/train.jsonl"  "$OUT_DIR/train.jsonl"
cp "$CORPUS_DIR/dev.jsonl"    "$OUT_DIR/dev.jsonl"
[[ -f "$CORPUS_DIR/test.jsonl" ]] && cp "$CORPUS_DIR/test.jsonl" "$OUT_DIR/test.jsonl"

echo ""
echo "Entraînement terminé."
echo "Modèle et données dans : $OUT_DIR"
