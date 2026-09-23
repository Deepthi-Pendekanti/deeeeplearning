# Project Description — Confidence-Aware GRU Virtual pH Sensor for Hydroponics

This project builds an **AI-driven virtual pH sensor for hydroponic (soil-less)
farming**, based on the research paper *"Toward sustainable hydroponic farming:
An AI-driven IoT framework for virtual pH sensing and sensor lifespan
extension"* (Moniruzzaman et al., Alexandria Engineering Journal, 2026) and its
publicly available Kaggle dataset (DOI 10.34740/kaggle/ds/8223904). In
hydroponics, plants grow directly in nutrient-rich water, and the water's pH
(acidity) must stay within a narrow range for the plants to absorb nutrients
properly. pH is normally measured with a physical electrochemical probe, but
these probes are fragile and expensive to maintain: they degrade with continuous
use, drift out of calibration, and need frequent replacement. The central idea
of the project is to reduce dependence on that physical probe by training a
machine-learning model to *predict* pH from the other, cheaper and more durable
sensors already present in the system — water temperature, air temperature,
humidity, total dissolved solids (nutrient concentration), and water level. This
predicted pH acts as a "virtual sensor," so the real probe can be read far less
often and therefore last much longer, lowering cost and maintenance while
keeping the system reliable.

Technically, the task is framed as **time-series forecasting**: using the
previous ten one-minute sensor readings to predict the next minute's pH value.
The raw dataset consists of roughly 50,570 readings recorded every ten seconds
over about a month. The project first cleans this data thoroughly — repairing
sensor-failure rows through linear interpolation, removing physically impossible
readings (such as negative nutrient levels or humidity above 100%), and
resampling the stream from ten-second to one-minute intervals. The cleaned data
is then normalized (scaled to a 0–1 range, with the scaler fit only on training
data to avoid contamination), converted into overlapping "windows" of ten time
steps, and split into training, validation, and test sets. The primary model is
a **Gated Recurrent Unit (GRU)**, a type of recurrent neural network designed
for sequential data, which the paper identifies as the best balance of accuracy
and computational efficiency. Alongside it, the project implements and compares
nine other models — LSTM and Transformer neural networks, the classical ARIMA
statistical method, Linear Regression, Random Forest, ExtraTrees, XGBoost, and a
trivial "persistence" baseline that simply predicts the last observed value — so
that the GRU's performance can be judged fairly against both simple and strong
alternatives.

The first phase of the work faithfully **reproduced the paper's results**,
achieving a coefficient of determination (R²) of 0.9824, matching the paper
almost exactly and confirming that the reproduction was correct down to the
model's exact parameter count of 24,351. However, the most important
contribution of the project came from **auditing that impressive result and
discovering it was inflated by temporal data leakage**. In the paper's
evaluation setup, the ten-minute windows were shuffled randomly into training
and test sets; because neighbouring windows overlap by nine of their ten minutes
and have almost-identical answers, the model was effectively being tested on
data nearly identical to what it had trained on. The project proved this by
showing that even the trivial persistence predictor also scored R²≈0.98 under
the same setup — a clear sign that the number reflected memorization rather than
genuine forecasting ability. To measure the model honestly, the project
introduced a **leakage-free temporal split**, in which the model trains on an
earlier time period and is tested on a genuinely later, unseen period, with a
gap between them so no test window overlaps any training window. Under this fair
test, the original GRU collapsed to R²=0.03 — worse than the dumb baseline —
revealing that the model had not actually learned to forecast pH at all.

The project then **redesigned the GRU to fix this honestly**. The key innovation
was a "delta head": instead of predicting the absolute pH value (which allowed
the network to cheat by echoing the last reading), the improved model predicts
the *change* in pH and adds it back to the last observed value, forcing it to
learn only the meaningful deviation from persistence. Combined with Huber loss
(which prevents a few glitchy sensor spikes from dominating training), dropout
and weight-decay regularization, a learning-rate scheduler, and early stopping,
this raised the honest R² from 0.03 to 0.65 — bringing the GRU up to match the
strongest baselines while using less than half the parameters (10,733 versus
24,051). An error analysis confirmed that the apparent gap between RMSE and MAE
was caused almost entirely by two physically-impossible pH spikes that were
uncaught sensor glitches, not model failures, and these were handled through
robust loss rather than by deleting data.

Beyond raw accuracy, the project delivered the features that make a virtual
sensor genuinely useful. It added a **confidence-aware hybrid sensing system**:
using Monte-Carlo dropout, the model estimates its own uncertainty at each step,
trusts its virtual prediction when confident, and calls the physical probe only
when uncertain or drifting. This measured a **75.6% reduction in physical-probe
usage** while maintaining accuracy — directly achieving the paper's stated goal
of extending sensor lifespan, but as a measured result rather than an
assumption. The project also **simulated realistic sensor failures** (missing
readings, frozen values, noise, drift, and calibration offsets across 30% of the
data stream) and showed that a traditional system blindly trusting the corrupted
probe collapses to R²=−6.9, whereas the proposed system detects the fault and
falls back to the virtual sensor, staying accurate at R²=0.89. Finally, the
model was profiled for **edge deployment**: it runs in about 0.9 milliseconds
per prediction on a plain CPU and exports to a 6.8 KB ONNX file with exact
numerical parity, making it suitable for low-power devices such as a Raspberry
Pi.

In summary, the project is not merely a reproduction that chases a high accuracy
number; it is a complete, honest, research-quality study. It reproduces the
published work, exposes a critical methodological flaw (data leakage) that
inflated the reported performance, corrects it with a principled model redesign,
and then demonstrates the real practical value of a virtual sensor through
measured sensor-usage reduction, fault robustness, and edge-readiness. The
entire pipeline is reproducible, every reported number is backed by a saved
experiment, and the work is transparent about its limitations — most notably
that on clean data the improved model matches rather than dramatically beats the
simple baselines, with its true advantages lying in reliability and efficiency
rather than raw accuracy.
