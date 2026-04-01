#!/usr/bin/env python3
"""
prepare_ocr.py — Étape 3 : ocr_pipeline
Concaténation et conversion des 8 fichiers JSONL (format {"breton": ..., "français": ...})
vers le format zeldarose ({"translation": {"br": ..., "fr": ...}}).
"""

import json
import pathlib
import random

# ── Chemins ──────────────────────────────────────────────────────────────────
CORPUS_DIR = pathlib.Path(__file__).parent.parent / "ocr_pipeline" / "corpus"
OUT = pathlib.Path(__file__).parent / "data" / "step3_ocr"
SEED = 42
DEV_RATIO = 0.10

# ── Lecture de tous les fichiers ──────────────────────────────────────────────
jsonl_files = sorted(CORPUS_DIR.glob("*.jsonl"))
print(f"Fichiers trouvés : {len(jsonl_files)}")
for f in jsonl_files:
    print(f"  - {f.name}")

raw_entries = []
for path in jsonl_files:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  {path.name}:{i} — JSONDecodeError : {e}")

print(f"Entrées brutes lues : {len(raw_entries)}")

# ── Conversion de format + nettoyage ─────────────────────────────────────────
def clean(s: str) -> str:
    return s.replace("\t", " ").strip()

valid = []
seen = set()
skipped = {"vide": 0, "format": 0, "doublon": 0}

for entry in raw_entries:
    # Gestion des deux formats possibles
    if "breton" in entry and "français" in entry:
        br = clean(entry["breton"])
        fr = clean(entry["français"])
    elif "translation" in entry:
        t = entry["translation"]
        br = clean(t.get("br", ""))
        fr = clean(t.get("fr", ""))
    else:
        skipped["format"] += 1
        continue

    if not br or not fr:
        skipped["vide"] += 1
        continue

    key = (br.lower(), fr.lower())
    if key in seen:
        skipped["doublon"] += 1
        continue
    seen.add(key)

    valid.append(json.dumps({"translation": {"br": br, "fr": fr}}, ensure_ascii=False))

print(f"Ignorées (vide) : {skipped['vide']}  |  (format) : {skipped['format']}  |  (doublon) : {skipped['doublon']}")
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
