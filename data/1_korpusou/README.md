# 1_korpusou — Corpus br↔fr agrégé (étape 1)

## Description

Corpus parallèle breton-français agrégé à partir de **9 sources** distinctes,
constituant la **première étape** du pipeline de fine-tuning M2M100.

Généré par le script `korpusou/tools/make_datasets.py`.

## Sources agrégées

| Corpus | Type |
|---|---|
| `jouitteau_translations` | Traductions manuelles |
| `Tatoeba` | Phrases parallèles collaboratives |
| `OfisPublik_merged` | Corpus de l'Office public de la langue bretonne |
| `Breton_KEB` | Corpus KEB |
| `Korpus_divyezhek_brezhoneg-galleg` | Corpus bilingue br↔fr |
| `ARBRES-Kenstur` | Phrases parallèles annotées |
| `apertium_test_corpus` | Corpus de test Apertium |
| `jouitteau_testcases` | Cas de test Jouitteau |
| `jouitteau_ofis_test` | Test Jouitteau / Ofis |

## Contenu

| Fichier | Lignes | Rôle |
|---|---|---|
| `train.jsonl` | 106 987 | Entraînement (90 %) |
| `dev.jsonl` | 5 943 | Validation zeldarose (5 %) |
| `tune.jsonl` | 5 943 | Fine-tuning complémentaire (5 %) |
| `test.jsonl` | 404 | Évaluation finale |
| `config.toml` | — | Configuration zeldarose pour cette étape |

## Format JSONL

```json
{
  "br": "...",
  "fr": "...",
  "split": "all",
  "origin": "OfisPublik_merged",
  "key": "hash_de_deduplication"
}
```

## Génération

```bash
cd korpusou/
python tools/make_datasets.py
```

Proportions effectives : `train=90%`, `dev=5%`, `tune=5%`
(filtrage et dédoublonnage appliqués avant le split)
