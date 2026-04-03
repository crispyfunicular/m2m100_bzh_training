# 2_termofis — TermOfis (pipeline en 3 étapes, archivé)

## Description

Données breton-français issues de la base de données terminologique **TermOfis**
(Office public de la langue bretonne).
Ces données constituaient la **deuxième étape** du pipeline de fine-tuning séquentiel,
désormais remplacé par l'approche unifiée (`data/unified/`).

## Contenu

| Fichier | Description |
|---|---|
| `train.jsonl` | ~79 300 paires terminologiques d'entraînement |
| `dev.jsonl` | ~8 800 paires de validation (split 90/10) |
| `config.toml` | Configuration zeldarose utilisée pour cette étape |

## Format JSONL

```json
{"translation": {"br": "...", "fr": "..."}}
```

## Source

Base TermOfis — équivalences terme-à-terme br↔fr.
Script de préparation : `prepare_termofis.py` (split 90/10, seed 42).

## Notes

- Modèle de départ : `output/step1_kenstur/` (résultat de l'étape 1)
- LR réduit à `3e-5` (continued fine-tuning, prévention de l'oubli catastrophique)
- Modèle résultant archivé dans : `output/step2_termofis/`
- Ce corpus est **écarté temporairement** du pipeline unifié et mis en réserve dans `data/terms/termofis.jsonl`.
