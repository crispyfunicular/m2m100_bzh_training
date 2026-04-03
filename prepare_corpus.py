#!/usr/bin/env python3
"""
prepare_corpus.py — Corpus unifié br↔fr pour le fine-tuning M2M100

Sources :
  1. ARBRES-Kenstur   — phrases parallèles (intégralement)
  2. TermOfis          — terminologie br↔fr (intégralement)
  3. ocr_pipeline      — lexiques historiques OCR :
       • len_br >= MIN_WORDS → phrase → entraînement
       • len_br <  MIN_WORDS → unité lexicale → data/terms/ocr_terms.jsonl

Sorties :
  data/unified/train.jsonl       — 90% du corpus (entraînement)
  data/unified/dev.jsonl         — 10% du corpus  (validation zeldarose)
  data/terms/termofis.jsonl      — TermOfis (écarté temporairement)
  data/terms/ocr_terms.jsonl     — unités lexicales OCR (non utilisées pour l'instant)
"""

import csv
import json
import pathlib
import random

# ── Paramètres ────────────────────────────────────────────────────────────────

# Seuil de longueur (tokens br) pour distinguer phrase et unité lexicale dans l'OCR
MIN_WORDS = 5

SEED      = 42
DEV_RATIO = 0.10

ROOT = pathlib.Path(__file__).parent.parent

# Sources
KENSTUR_FILE  = ROOT / "korpusou"    / "corpora" / "ARBRES-Kenstur" / "data" / "train.jsonl"
TERMOFIS_FILE = ROOT / "termofis"    / "src"     / "TermOfis.csv"
OCR_DIR       = ROOT / "ocr_pipeline" / "corpus"

# Sorties
OUT_UNIFIED = pathlib.Path(__file__).parent / "data" / "unified"
OUT_TERMS   = pathlib.Path(__file__).parent / "data" / "terms"


# ── Utilitaires ───────────────────────────────────────────────────────────────

def clean(s: str) -> str:
    return s.replace("\t", " ").strip()


def to_jsonl(br: str, fr: str) -> str:
    return json.dumps({"translation": {"br": br, "fr": fr}}, ensure_ascii=False)


# ── 1. ARBRES-Kenstur ─────────────────────────────────────────────────────────

kenstur_lines = []
seen: set[tuple[str, str]] = set()

for line in KENSTUR_FILE.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)["translation"]
        br, fr = clean(d["br"]), clean(d["fr"])
        if br and fr and (br.lower(), fr.lower()) not in seen:
            seen.add((br.lower(), fr.lower()))
            kenstur_lines.append(to_jsonl(br, fr))
    except (KeyError, json.JSONDecodeError):
        pass

print(f"Kenstur          : {len(kenstur_lines):>7} entrées")


# ── 2. TermOfis ───────────────────────────────────────────────────────────────

termofis_lines = []
termofis_skipped = {"vide": 0, "doublon": 0}

with TERMOFIS_FILE.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        fr_raw = row.get("Penntermen fra", row.get("penntermen fra", ""))
        br_raw = row.get("Penntermen bre", row.get("penntermen bre", ""))
        br, fr = clean(br_raw), clean(fr_raw)
        if not br or not fr:
            termofis_skipped["vide"] += 1
            continue
        key = (br.lower(), fr.lower())
        if key in seen:
            termofis_skipped["doublon"] += 1
            continue
        seen.add(key)
        termofis_lines.append(to_jsonl(br, fr))

print(f"TermOfis         : {len(termofis_lines):>7} entrées → data/terms/termofis.jsonl (écarté temporairement)"
      f"  (ignorées : vides={termofis_skipped['vide']}, doublons={termofis_skipped['doublon']})")


# ── 3. OCR pipeline ───────────────────────────────────────────────────────────

ocr_sentences = []
ocr_terms = []
ocr_stats = {"vide": 0, "format": 0, "doublon": 0}

jsonl_files = sorted(OCR_DIR.glob("*.jsonl"))
print(f"OCR fichiers     : {len(jsonl_files)}")

for path in jsonl_files:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                ocr_stats["format"] += 1
                continue

            if "breton" in entry and "français" in entry:
                br, fr = clean(entry["breton"]), clean(entry["français"])
            elif "translation" in entry:
                t = entry["translation"]
                br, fr = clean(t.get("br", "")), clean(t.get("fr", ""))
            else:
                ocr_stats["format"] += 1
                continue

            if not br or not fr:
                ocr_stats["vide"] += 1
                continue

            key = (br.lower(), fr.lower())
            if key in seen:
                ocr_stats["doublon"] += 1
                continue
            seen.add(key)

            if len(br.split()) >= MIN_WORDS:
                ocr_sentences.append(to_jsonl(br, fr))
            else:
                ocr_terms.append(to_jsonl(br, fr))

print(f"OCR → phrases    : {len(ocr_sentences):>7}  (len_br >= {MIN_WORDS})")
print(f"OCR → unités lex.: {len(ocr_terms):>7}  → data/terms/ocr_terms.jsonl"
      f"  (ignorées : vides={ocr_stats['vide']}, format={ocr_stats['format']}, doublons={ocr_stats['doublon']})")


# ── Aperçu pour calibrage du seuil MIN_WORDS ─────────────────────────────────

print()
print(f"── 10 exemples de phrases OCR retenues (len_br >= {MIN_WORDS}) ─────────────")
rng_preview = random.Random(SEED)
for ex in rng_preview.sample(ocr_sentences, min(10, len(ocr_sentences))):
    d = json.loads(ex)["translation"]
    print(f"  br ({len(d['br'].split()):2} mots) : {d['br'][:80]}")
    print(f"  fr              : {d['fr'][:80]}")
    print()

print(f"── 10 exemples d'unités lexicales écartées (len_br < {MIN_WORDS}) ──────────")
for ex in rng_preview.sample(ocr_terms, min(10, len(ocr_terms))):
    d = json.loads(ex)["translation"]
    print(f"  br ({len(d['br'].split()):2} mot(s)) : {d['br'][:80]}")
    print(f"  fr              : {d['fr'][:80]}")
    print()


# ── Assemblage + shuffle + split ──────────────────────────────────────────────

# TermOfis écarté temporairement → data/terms/termofis.jsonl
all_train = kenstur_lines + ocr_sentences
rng = random.Random(SEED)
rng.shuffle(all_train)

n_dev = max(1, round(len(all_train) * DEV_RATIO))
dev_lines   = all_train[:n_dev]
train_lines = all_train[n_dev:]

print("─" * 55)
print(f"Total entraînement : {len(all_train):>7}")
print(f"  Train            : {len(train_lines):>7}")
print(f"  Dev              : {len(dev_lines):>7}")
print()


# ── Écriture ──────────────────────────────────────────────────────────────────

OUT_UNIFIED.mkdir(parents=True, exist_ok=True)
OUT_TERMS.mkdir(parents=True, exist_ok=True)

(OUT_UNIFIED / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
(OUT_UNIFIED / "dev.jsonl").write_text("\n".join(dev_lines) + "\n", encoding="utf-8")
(OUT_TERMS / "ocr_terms.jsonl").write_text("\n".join(ocr_terms) + "\n", encoding="utf-8")
(OUT_TERMS / "termofis.jsonl").write_text("\n".join(termofis_lines) + "\n", encoding="utf-8")

print(f"✅ data/unified/train.jsonl    ({len(train_lines)} lignes)")
print(f"✅ data/unified/dev.jsonl      ({len(dev_lines)} lignes)")
print(f"✅ data/terms/ocr_terms.jsonl  ({len(ocr_terms)} lignes)")
print(f"✅ data/terms/termofis.jsonl   ({len(termofis_lines)} lignes)  [écarté temporairement]")
