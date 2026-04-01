#!/usr/bin/env bash
# train.sh — Pipeline complet de fine-tuning m2m100 en 3 étapes (br↔fr)
#
# Prérequis : pip install zeldarose sentencepiece
#
# Usage : bash train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════"
echo "  Préparation des données"
echo "════════════════════════════════════════════════"

python3 prepare_kenstur.py
python3 prepare_termofis.py
python3 prepare_ocr.py

echo ""
echo "════════════════════════════════════════════════"
echo "  Validation des données"
echo "════════════════════════════════════════════════"

python3 - <<'EOF'
import json, pathlib
for step in ['step1_kenstur', 'step2_termofis', 'step3_ocr']:
    for split in ['train', 'dev']:
        p = pathlib.Path(f'data/{step}/{split}.jsonl')
        n = 0
        for line in p.open():
            d = json.loads(line)['translation']
            assert d['br'].strip() and d['fr'].strip(), f"Champ vide dans {step}/{split}"
            n += 1
        print(f'  {step}/{split}: {n} lignes OK')
EOF

echo ""
echo "════════════════════════════════════════════════"
echo "  Étape 1 : ARBRES-Kenstur (phrases parallèles)"
echo "════════════════════════════════════════════════"

zeldarose transformer \
    --tokenizer facebook/m2m100_418M \
    --pretrained-model facebook/m2m100_418M \
    --config config_step1_kenstur.toml \
    --out-dir output/step1_kenstur \
    --val-text data/step1_kenstur/dev.jsonl \
    data/step1_kenstur/train.jsonl

echo ""
echo "════════════════════════════════════════════════"
echo "  Étape 2 : TermOfis (terminologie)"
echo "════════════════════════════════════════════════"

zeldarose transformer \
    --tokenizer facebook/m2m100_418M \
    --pretrained-model output/step1_kenstur \
    --config config_step2_termofis.toml \
    --out-dir output/step2_termofis \
    --val-text data/step2_termofis/dev.jsonl \
    data/step2_termofis/train.jsonl

echo ""
echo "════════════════════════════════════════════════"
echo "  Étape 3 : ocr_pipeline (lexiques historiques)"
echo "════════════════════════════════════════════════"

zeldarose transformer \
    --tokenizer facebook/m2m100_418M \
    --pretrained-model output/step2_termofis \
    --config config_step3_ocr.toml \
    --out-dir output/step3_ocr \
    --val-text data/step3_ocr/dev.jsonl \
    data/step3_ocr/train.jsonl

echo ""
echo "✅ Pipeline complet terminé. Modèle final dans output/step3_ocr/"
