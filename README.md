# Retrofit Forecast ML Pipeline

End-to-end machine-learning pipeline for predicting automotive retrofit visits.
The pipeline covers data cleansing, feature engineering, binary/multiclass
classification, temporal regression, time-series forecasting, AutoML
benchmarking, and a combined forecast with backtesting.

## Project Structure

```
code_paper/
├── run_pipeline.py              # Entry point – runs the full 8-step pipeline
├── src/
│   ├── config.py                # All constants, paths, and hyperparameters
│   ├── utils.py                 # Checkpointing, memory tracking, validation
│   ├── visualization.py         # Shared plotting helpers
│   ├── 01_data_cleansing_and_transformation.py
│   ├── 02_feature_engineering.py
│   ├── 03_data_modeling_and_evaluation_binary.py
│   ├── 04_data_modeling_and_evaluation_multiclass.py
│   ├── 05_data_modeling_and_evaluation_temporal.py
│   ├── 06_data_modeling_and_evaluation_time_series.py
│   ├── 07_data_modeling_and_evaluation_automl.py
│   └── 08_complete_forecast.py
```

## Data

Place the two anonymised CSV files in the **parent folder** (`paper/`):

```
paper/
├── anonymized_data_retrofits.csv    # Source A – retrofit orders
├── anonymized_data_prototypes.csv   # Source B – vehicle master data
└── code_paper/                      # ← this repository
```

The CSVs use **semicolons (`;`)** as delimiter. Column names are mapped
automatically on load (see `config.CSV_RENAME_SOURCE_A` /
`CSV_RENAME_SOURCE_B`).

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 01 | `01_data_cleansing_and_transformation` | Load CSVs, clean dates, normalise types, merge sources |
| 02 | `02_feature_engineering` | Auto-detect features, encode categoricals, create train/test splits |
| 03 | `03_data_modeling_and_evaluation_binary` | Binary classification – will a vehicle come for retrofit? |
| 04 | `04_data_modeling_and_evaluation_multiclass` | Multiclass classification – which retrofit type? |
| 05 | `05_data_modeling_and_evaluation_temporal` | Temporal regression – how long will the retrofit take? |
| 06 | `06_data_modeling_and_evaluation_time_series` | Time-series forecasting – monthly aggregate volume |
| 07 | `07_data_modeling_and_evaluation_automl` | AutoML benchmark (AutoGluon, TabPFN) vs manual models |
| 08 | `08_complete_forecast` | Combined forecast, backtest, calibration, CSV export |

Checkpoints are saved to `.cache/` so interrupted runs resume automatically.

## Installation

### Required (core pipeline)

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy \
            xgboost lightgbm catboost imbalanced-learn psutil joblib optuna shap
```

### Optional (foundation models and AutoML)

These are auto-installed on first run if missing. Install manually for
faster startup:

```bash
pip install torch transformers              # TIME-LLM backbone
pip install chronos-forecasting             # Amazon Chronos time-series
pip install statsmodels prophet             # Classical time-series (ARIMA, Holt-Winters)
pip install tabpfn                          # TabPFN foundation model
pip install "autogluon.tabular[all]"        # AutoGluon AutoML
pip install boto3                           # Amazon Bedrock LLM integration
```

> **Note:** `boto3` is only needed if you have access to Amazon Bedrock for
> LLM-based models (LLM-Embedding, LLM-Prompted, LLM-Stacked). The pipeline
> gracefully skips these models when `boto3` is unavailable.

## Usage

```bash
cd code_paper
python run_pipeline.py
```

All configuration (hyperparameters, thresholds, column names) lives in
`src/config.py`. The output forecast is written to
`paper/forecast_output.csv`.
