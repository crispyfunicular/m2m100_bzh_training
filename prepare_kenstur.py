#!/usr/bin/env python3
"""
prepare_kenstur.py — Étape 1 : ARBRES-Kenstur
Split 90/10 du corpus de phrases parallèles br↔fr.
"""

import json
import pathlib
import random

# ── Chemins ──────────────────────────────────────────────────────────────────
SRC = pathlib.Path(__file__).parent.parent / "korpusou" / "corpora" / "ARBRES-Kenstur" / "data" / "train.jsonl"
OUT = pathlib.Path(__file__).parent / "data" / "step1_kenstur"
SEED = 42
DEV_RATIO = 0.10

# ── Lecture ───────────────────────────────────────────────────────────────────
lines = SRC.read_text(encoding="utf-8").splitlines()
lines = [l for l in lines if l.strip()]
print(f"Lignes lues : {len(lines)}")

# ── Validation du format ──────────────────────────────────────────────────────
valid = []
for i, line in enumerate(lines, 1):
    try:
        d = json.loads(line)
        t = d["translation"]
        assert t["br"].strip() and t["fr"].strip(), f"Champ vide ligne {i}"
        valid.append(line)
    except (KeyError, AssertionError, json.JSONDecodeError) as e:
        print(f"  ⚠️  Ligne {i} ignorée : {e}")

print(f"Lignes valides : {len(valid)}")

# ── Shuffle + split ───────────────────────────────────────────────────────────
rng = random.Random(SEED)
rng.shuffle(valid)

n_dev = max(1, round(len(valid) * DEV_RATIO))
dev_lines = valid[:n_dev]
train_lines = valid[n_dev:]

print(f"Train : {len(train_lines)}  |  Dev : {len(dev_lines)}")

# ── Écriture ──────────────────────────────────────────────────────────────────
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
(OUT / "dev.jsonl").write_text("\n".join(dev_lines) + "\n", encoding="utf-8")

print(f"✅ Données écrites dans {OUT}")
