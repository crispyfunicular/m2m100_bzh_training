#!/usr/bin/env python3
"""
prepare_termofis.py — Étape 2 : TermOfis
Conversion CSV (séparateur ;) → JSONL zeldarose, nettoyage, dédoublonnage, split 90/10.
Colonnes attendues : Tachenn ; Penntermen fra ; Penntermen bre ; RG bre
"""

import csv
import json
import pathlib
import random

# ── Chemins ──────────────────────────────────────────────────────────────────
SRC = pathlib.Path(__file__).parent.parent / "TermOfis" / "src" / "TermOfis.csv"
OUT = pathlib.Path(__file__).parent / "data" / "step2_termofis"
SEED = 42
DEV_RATIO = 0.10

# ── Lecture CSV ───────────────────────────────────────────────────────────────
raw_rows = []
with SRC.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        raw_rows.append(row)

print(f"Lignes CSV lues : {len(raw_rows)}")
print(f"Colonnes : {list(raw_rows[0].keys()) if raw_rows else []}")

# ── Extraction et nettoyage ───────────────────────────────────────────────────
def clean(s: str) -> str:
    """Supprime tabs parasites et espaces superflus."""
    return s.replace("\t", " ").strip()

valid = []
seen = set()
skipped = {"vide": 0, "doublon": 0}

for row in raw_rows:
    # Noms de colonnes du CSV TermOfis
    fr_raw = row.get("Penntermen fra", row.get("penntermen fra", ""))
    br_raw = row.get("Penntermen bre", row.get("penntermen bre", ""))

    fr = clean(fr_raw)
    br = clean(br_raw)

    if not fr or not br:
        skipped["vide"] += 1
        continue

    key = (br.lower(), fr.lower())
    if key in seen:
        skipped["doublon"] += 1
        continue
    seen.add(key)

    valid.append(json.dumps({"translation": {"br": br, "fr": fr}}, ensure_ascii=False))

print(f"Ignorées (vide) : {skipped['vide']}  |  (doublon) : {skipped['doublon']}")
print(f"Entrées valides : {len(valid)}")

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
