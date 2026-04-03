# 3_ocr — Lexiques OCR (pipeline en 3 étapes, archivé)

## Description

Données breton-français extraites par OCR depuis des **lexiques historiques** numérisés
(dictionnaires Le Gonidec 1919, Bozec 1933, Normant 1902, etc.).
Ces données constituaient la **troisième étape** du pipeline de fine-tuning séquentiel,
désormais remplacé par l'approche unifiée (`data/unified/`).

## Contenu

| Fichier | Description |
|---|---|
| `train.jsonl` | ~21 300 paires d'entraînement (toutes entrées confondues) |
| `dev.jsonl` | ~2 370 paires de validation (split 90/10) |
| `config.toml` | Configuration zeldarose utilisée pour cette étape |

## Format JSONL

```json
{"translation": {"br": "...", "fr": "..."}}
```

## Source

Pipeline `ocr_pipeline/corpus/*.jsonl` — entrées issues de dictionnaires bilingues br↔fr.
Script de préparation : `prepare_ocr.py` (concaténation + dédoublonnage + split 90/10).

## Tri dans le pipeline unifié

Le corpus OCR brut contient un mélange de :
- **Unités lexicales** (1–4 tokens br) → ~93,6 % des entrées → `data/terms/ocr_terms.jsonl`
- **Phrases** (≥ 5 tokens br) → ~6,4 % des entrées → incluses dans `data/unified/`

## Notes

- Modèle de départ : `output/step2_termofis/` (résultat de l'étape 2)
- LR réduit à `2e-5` (3ème étape de continued fine-tuning)
- Modèle résultant archivé dans : `output/step3_ocr/`
