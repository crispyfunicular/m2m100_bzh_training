# Fine-tuning m2m100 par étapes successives (br↔fr)

## 1. Objectif

Fine-tuner progressivement le modèle [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M) sur trois corpus breton-français de nature complémentaire, en reprenant à chaque étape les poids de la précédente (*continued fine-tuning*).

L'entraînement utilise [**zeldarose**](https://zeldarose.readthedocs.io/en/latest/) (Grobol, EMNLP 2023), outil CLI pour l'entraînement de transformers, avec sa tâche `mbart` compatible m2m100.

## 2. Vue d'ensemble des étapes

| Étape | Corpus | Nature | Volume | Format actuel |
|:---:|---|---|---:|---|
| 1 | ARBRES-Kenstur | Phrases parallèles | 7 543 | ✅ zeldarose natif |
| 2 | TermOfis | Termes / terminologie | ~88 000 | ⚠️ tabs à nettoyer |
| 3 | ocr_pipeline | Lexiques historiques OCR | 25 031 | ❌ conversion requise |
| | **Total** | | **~120 000** | |

**Logique de séquençage** : on commence par les phrases complètes (domaine le plus proche du pré-entraînement de m2m100), puis on spécialise vers la terminologie, et enfin on enrichit avec le vocabulaire historique extrait par OCR.

## 3. Infrastructure

| Ressource | Détail |
|---|---|
| **GPU** | Machine distante, 32 Go VRAM |
| **Modèle de base** | `facebook/m2m100_418M` (~1.8 Go) |
| **Dépendances** | `zeldarose`, `sentencepiece` |
| **Arborescence projet** | `~/git/modyco/breton/m2m100_training/` |

## 4. Préparation des données

Chaque corpus nécessite un script de conversion produisant `train.jsonl` + `dev.jsonl` au format :
```json
{"translation": {"br": "skiant", "fr": "science"}}
```

### 4.1 Étape 1 — ARBRES-Kenstur

**Source** : `korpusou/corpora/ARBRES-Kenstur/data/train.jsonl`

Déjà au bon format. Seul un split train/dev est à produire (le fichier actuel ne contient qu'un `train.jsonl`).

**Script** : `prepare_kenstur.py`
- Split 90/10, seed fixe
- Sortie dans `data/step1_kenstur/`

### 4.2 Étape 2 — TermOfis

**Source** : `TermOfis/src/TermOfis.csv` (CSV `;`)

**Script** : `prepare_termofis.py`
- Lecture CSV, renommage colonnes
- Nettoyage des tabs parasites (61 entrées) et des espaces
- Dédoublonnage
- Split 90/10, seed fixe
- Sortie dans `data/step2_termofis/`

### 4.3 Étape 3 — ocr_pipeline

**Source** : `ocr_pipeline/corpus/*.jsonl` (8 fichiers)

> [!WARNING]
> Format différent : `{"breton": "...", "français": "..."}` au lieu du format zeldarose. Conversion nécessaire.

**Script** : `prepare_ocr.py`
- Lecture et concaténation des 8 fichiers
- Conversion des clés `breton` → `br`, `français` → `fr`, encapsulation dans `{"translation": {...}}`
- Nettoyage et dédoublonnage
- Split 90/10, seed fixe
- Sortie dans `data/step3_ocr/`

## 5. Configurations d'entraînement

Trois fichiers TOML, un par étape. Les hyperparamètres varient selon la nature des données.

### 5.1 `config_step1_kenstur.toml` — Phrases

```toml
type = "mbart"

[task]
denoise_loss_ratio = 0.0
source_langs = ["br"]
target_langs = ["fr"]
strict_langs = false

[tuning]
batch_size = 32               # phrases longues → batch plus petit
learning_rate = 5e-5
gradient_clipping = 1.0
max_input_length = 256        # phrases complètes
max_epochs = 20               # corpus petit → plus d'epochs
warmup_steps = 200
```

### 5.2 `config_step2_termofis.toml` — Terminologie

```toml
type = "mbart"

[task]
denoise_loss_ratio = 0.0
source_langs = ["br"]
target_langs = ["fr"]
strict_langs = false

[tuning]
batch_size = 64               # termes courts → gros batch
learning_rate = 3e-5          # LR réduit (continued fine-tuning)
gradient_clipping = 1.0
max_input_length = 128        # termes courts
max_epochs = 10
warmup_steps = 500
```

### 5.3 `config_step3_ocr.toml` — Lexiques historiques

```toml
type = "mbart"

[task]
denoise_loss_ratio = 0.0
source_langs = ["br"]
target_langs = ["fr"]
strict_langs = false

[tuning]
batch_size = 64
learning_rate = 2e-5          # LR encore réduit (3ème étape)
gradient_clipping = 1.0
max_input_length = 128
max_epochs = 10
warmup_steps = 300
```

> [!NOTE]
> Le learning rate diminue à chaque étape (`5e-5` → `3e-5` → `2e-5`) pour éviter le *catastrophic forgetting* des connaissances acquises à l'étape précédente.

## 6. Procédure de lancement

```bash
# 0. Installation
pip install zeldarose sentencepiece

# 1. Préparer toutes les données
python3 prepare_kenstur.py
python3 prepare_termofis.py
python3 prepare_ocr.py

# 2. Étape 1 : ARBRES-Kenstur (phrases)
zeldarose transformer \
    --tokenizer facebook/m2m100_418M \
    --pretrained-model facebook/m2m100_418M \
    --config config_step1_kenstur.toml \
    --out-dir output/step1_kenstur \
    --val-text data/step1_kenstur/dev.jsonl \
    data/step1_kenstur/train.jsonl

# 3. Étape 2 : TermOfis (terminologie)
#    → reprend les poids de l'étape 1
zeldarose transformer \
    --tokenizer facebook/m2m100_418M \
    --pretrained-model output/step1_kenstur \
    --config config_step2_termofis.toml \
    --out-dir output/step2_termofis \
    --val-text data/step2_termofis/dev.jsonl \
    data/step2_termofis/train.jsonl

# 4. Étape 3 : ocr_pipeline (lexiques historiques)
#    → reprend les poids de l'étape 2
zeldarose transformer \
    --tokenizer facebook/m2m100_418M \
    --pretrained-model output/step2_termofis \
    --config config_step3_ocr.toml \
    --out-dir output/step3_ocr \
    --val-text data/step3_ocr/dev.jsonl \
    data/step3_ocr/train.jsonl
```

## 7. Arborescence du projet

```
m2m100_training/
├── prepare_kenstur.py
├── prepare_termofis.py
├── prepare_ocr.py
├── config_step1_kenstur.toml
├── config_step2_termofis.toml
├── config_step3_ocr.toml
├── train.sh
├── data/
│   ├── step1_kenstur/     (train.jsonl, dev.jsonl)
│   ├── step2_termofis/    (train.jsonl, dev.jsonl)
│   └── step3_ocr/         (train.jsonl, dev.jsonl)
└── output/
    ├── step1_kenstur/     (modèle après étape 1)
    ├── step2_termofis/    (modèle après étape 2)
    └── step3_ocr/         (modèle final)
```

## 8. Vérification

### 8.1 Validation des données (avant chaque étape)

```bash
python3 -c "
import json, pathlib
for step in ['step1_kenstur', 'step2_termofis', 'step3_ocr']:
    for split in ['train', 'dev']:
        p = pathlib.Path(f'data/{step}/{split}.jsonl')
        n = 0
        for line in p.open():
            d = json.loads(line)['translation']
            assert d['br'].strip() and d['fr'].strip()
            n += 1
        print(f'{step}/{split}: {n} lignes OK')
"
```

### 8.2 Test du modèle final

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("output/step3_ocr")
tokenizer = M2M100Tokenizer.from_pretrained("output/step3_ocr")
tokenizer.src_lang = "br"

tests = ["skiant", "Me am eus kanet", "kelenn"]
for t in tests:
    inputs = tokenizer(t, return_tensors="pt")
    gen = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("fr"))
    print(f"  {t} → {tokenizer.decode(gen[0], skip_special_tokens=True)}")
```
