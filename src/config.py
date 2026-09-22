"""
Central configuration for the GRU virtual pH sensor reproduction.

Every value here is traceable to the paper:
"Toward sustainable hydroponic farming: An AI-driven IoT framework for
 virtual pH sensing and sensor lifespan extension"
Moniruzzaman et al., Alexandria Engineering Journal 143 (2026) 40-63.

Section references are given inline.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
METRICS = RESULTS / "metrics"

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, MODELS, METRICS):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Dataset (Data availability section)
# ----------------------------------------------------------------------------
KAGGLE_DATASET_DOI = "10.34740/kaggle/ds/8223904"
# Kaggle API "owner/slug" is resolved at download time (see download_data.py).

# ----------------------------------------------------------------------------
# Preprocessing (Section 3.3.5 / 3.3.6 / 3.3.10)
# ----------------------------------------------------------------------------
RAW_INTERVAL_SECONDS = 10          # RPi records every 10 s (Sec 3.3.5)
RESAMPLE_INTERVAL = "1min"         # resampled/interpolated to 1 min (Sec 3.3.6 / 3.3.10)
IS_DEFAULT_COLUMN = "isDefault"    # sensor-failure flag == 1 (Sec 3.3.5)

# ----------------------------------------------------------------------------
# Features (Section 3.3.7 + Appendix A.1.2)
# The final GRU virtual sensor uses SENSOR-ONLY inputs.
# Appendix A.1.2 lists exactly 5 input features:
#   1. pH (current value)  2. TDS  3. DHT humidity  4. Water temperature  5. Water level
# ----------------------------------------------------------------------------
TARGET = "pH"
# Canonical feature names used internally after column normalization.
#
# NOTE on feature count: The paper is internally inconsistent about the exact
# input feature set for the final GRU:
#   * Appendix A.1.2 lists 5 inputs (pH, TDS, DHT_humidity, water_temp, water_level).
#   * Table A.9 states the input layer is (batch, 10, 8) and the model has 24,351
#     trainable params -- which corresponds EXACTLY to 8 input features.
#   * The ablation (Table 8) reports the best GRU (RMSE 0.0181) under "All sensors".
# The full sensor suite in the dataset is 6 sensors:
#     pH, TDS, water_level, DHT_temp, DHT_humidity, water_temp
# We treat "All sensors" (these 6) as the primary sensor-only configuration
# (this matches Table 8's winning label), and we additionally expose an 8-feature
# variant that reproduces the exact 24,351-parameter count from Table A.9.
SENSOR_FEATURES = [
    "pH", "TDS", "water_level", "DHT_temp", "DHT_humidity", "water_temp",
]
# Actuator columns (used for the ablation study, Section 4.4).
ACTUATOR_FEATURES = ["pH_reducer", "add_water", "nutrients_adder", "humidifier", "ex_fan"]
# 8-feature set that matches Table A.9's (batch,10,8) input and 24,351 params
# (6 sensors + the two most pH-relevant actuators: pH_reducer, add_water).
SENSOR_FEATURES_8 = SENSOR_FEATURES + ["pH_reducer", "add_water"]

# ----------------------------------------------------------------------------
# Windowing (Section 3.3.7 / 3.3.10 / Appendix A.1.2)
# ----------------------------------------------------------------------------
LOOKBACK = 10          # 10 historical values y_{t-10}..y_{t-1} (Sec 3.3.7)
HORIZON = 1            # one-step-ahead pH (t+1) (Sec 3.3.10)

# ----------------------------------------------------------------------------
# Train/test split (Section 3.3.10)
# ----------------------------------------------------------------------------
TRAIN_RATIO = 0.80     # chronological 80:20 split
VAL_RATIO = 0.10       # validation subset drawn from the training portion

# ----------------------------------------------------------------------------
# Training hyperparameters (Section 3.3.10)
# ----------------------------------------------------------------------------
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3   # Adam
OPTIMIZER = "adam"
LOSS = "mse"
N_RUNS = 3             # "All models were trained three times"
SEEDS = [42, 43, 44]   # three deterministic seeds for the three runs

# ----------------------------------------------------------------------------
# Model architectures (Table 4 and Table A.9)
# ----------------------------------------------------------------------------
# GRU: GRU(50, return_sequences=True) -> Dropout -> GRU(50) -> Dropout -> Dense(1)
GRU_UNITS = 50
# Table 4 says dropout 0.1 per layer; Table A.9 says 0.2. We expose both and
# default to Table A.9 (the dedicated GRU architecture appendix). See README.
GRU_DROPOUT = 0.2

# LSTM: LSTM(43, RS=True) -> LSTM(43) -> Dense(1), 0.1 dropout per layer.
LSTM_UNITS = 43
LSTM_DROPOUT = 0.1

# Transformer: encoder-decoder, 3 layers, 4 heads, d_model=64, dropout 0.1.
TRANSFORMER_D_MODEL = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 3
TRANSFORMER_DROPOUT = 0.1

# ----------------------------------------------------------------------------
# Physical validity ranges (Algorithm 2 "ValidateAndFilter" range check, Sec 3.2.4)
# Readings outside these physically-plausible bounds are treated as sensor
# faults (like isDefault==1) and repaired via the same linear interpolation.
# Bounds are generous physical limits, NOT the crop setpoints, so we do not
# bias the data toward the target distribution.
# ----------------------------------------------------------------------------
VALID_RANGES = {
    "pH": (0.0, 14.0),            # chemical pH scale
    "TDS": (0.0, 3000.0),         # ppm, non-negative; well above nutrient max
    "water_level": (0.0, 10.0),   # discrete level sensor
    "DHT_temp": (0.0, 60.0),      # DHT22 operating range
    "DHT_humidity": (0.0, 100.0), # relative humidity is a percentage
    "water_temp": (0.0, 40.0),    # DS18B20 realistic aquatic range
}

# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
RESIDUAL_TOLERANCE = 0.05   # +/-0.05 pH tolerance band (Sec 3.3.2 / 4.5)

# Paper's reported GRU targets (Table 5 / Table 6) for comparison.
PAPER_TARGETS = {
    "GRU": {"RMSE": 0.0181, "MAE": 0.0116, "R2": 0.9824},
    "LSTM": {"RMSE": 0.0190, "MAE": 0.0124, "R2": 0.9787},
    "Transformer": {"RMSE": 0.0206, "MAE": 0.0128, "R2": 0.9749},
}
