# Garbage Image Classification

Deep-learning waste classifier trained on a combined dataset from 3 Kaggle sources
(trashnet, garbage-classification 12-class, garbage-classification-v2), unified into
7 categories: **cardboard, glass, metal, paper, plastic, trash, biological**.

## Project layout

```
trashnet/, garbage-classification/, garbage-classification-v2/   # raw downloaded datasets
data/unified/<class>/                # consolidated, deduped, class-mapped images
data/splits/{train,val,test}/<class>/  # stratified 70/15/15 split
src/data_prep/                       # class mapping, consolidation, EDA, splitting
src/training/                        # tf.data pipelines, model builders, training loop
src/evaluation/                      # metrics, confusion matrices, best-model selection
models/<model_key>/                  # per-model checkpoint + history + test metrics
models/best_model/                   # winning model + label_map.json, used by the app
reports/eda/                         # class distribution, sample grid, pixel intensity plots
reports/evaluation/                  # per-class metrics, confusion matrices, comparison table
app/streamlit_app.py                 # upload-and-predict UI
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `tensorflow-macos` + `tensorflow-metal` for Apple Silicon GPU
acceleration. **For Streamlit Cloud or any non-macOS deployment**, swap those two lines
for plain `tensorflow` before deploying.

## Reproducing the pipeline end to end

All commands assume the venv above is activated and the working directory is the
project root, unless noted.

1. **Consolidate the 3 raw datasets** into `data/unified/<class>/` (dedupes exact
   duplicates across sources, drops corrupt/unsupported image files, drops
   battery/clothes/shoes which aren't part of the unified taxonomy):
   ```bash
   cd src/data_prep && python3 consolidate_datasets.py
   ```

2. **Run EDA** (writes plots to `reports/eda/`):
   ```bash
   python3 eda.py
   ```

3. **Split into train/val/test** (stratified 70/15/15, `data/splits/`):
   ```bash
   python3 split_dataset.py
   ```

4. **Train each model** (from `src/training/`):
   ```bash
   cd ../training
   python3 train.py baseline_cnn      --epochs 20 --fine-tune-epochs 0
   python3 train.py mobilenetv2       --epochs 15 --fine-tune-epochs 8
   python3 train.py efficientnetb0    --epochs 15 --fine-tune-epochs 8
   ```
   Each run trains with a frozen ImageNet backbone first (transfer-learning models
   only), then unfreezes the top ~20% of backbone layers for a short fine-tuning
   pass at a lower learning rate. Class imbalance is handled via `class_weight`.
   Outputs land in `models/<model_key>/`: `model.keras`, `history.json`,
   `class_names.json`, `test_metrics.json`.

5. **Evaluate all trained models** (from `src/evaluation/`):
   ```bash
   cd ../evaluation
   python3 evaluate.py
   ```
   Writes per-class precision/recall/F1 CSVs and confusion-matrix heatmaps to
   `reports/evaluation/`, plus `comparison_table.csv` ranking all models by macro F1.

6. **Select and stage the best model**:
   ```bash
   python3 select_best_model.py
   ```
   Copies the highest-macro-F1 model into `models/best_model/` along with
   `label_map.json` and `model_info.json` (used by the Streamlit app).

7. **Run the app** (from the project root):
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Upload an image; the app shows the predicted class and a top-3 confidence
   breakdown.

## Deployment

- **Local**: `streamlit run app/streamlit_app.py` as above.
- **Streamlit Community Cloud**: push this repo (swap `tensorflow-macos`/
  `tensorflow-metal` for `tensorflow` in `requirements.txt` first), connect the repo
  on share.streamlit.io, and point it at `app/streamlit_app.py`. `models/best_model/`
  must be committed (or fetched at startup) since the app loads it directly from disk.
