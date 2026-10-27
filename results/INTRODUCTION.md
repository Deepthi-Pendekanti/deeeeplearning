# Introduction

## Background and motivation

Agriculture is under mounting pressure from shrinking arable land, climate
variability, and a growing global population expected to reach roughly 9.7
billion by 2050. These constraints have accelerated interest in **soil-less
cultivation methods**, among which **hydroponics** — growing plants directly in
a nutrient-rich water solution rather than soil — has emerged as a leading
approach. Hydroponics offers precise control over the plant's root environment,
higher yields per unit area, efficient water use, and suitability for vertical
and indoor farming. However, this precision comes at the cost of constant,
careful monitoring: because the roots sit directly in water, even small
deviations in the solution's chemistry can quickly harm the crop.

The single most critical chemical parameter in a hydroponic system is the **pH**
(acidity) of the nutrient solution. pH governs how readily plants can absorb
dissolved nutrients; if it drifts outside the crop-appropriate band (typically
around 5.5–6.5), nutrients become chemically unavailable and growth suffers even
when nutrients are physically present. Consequently, hydroponic systems
continuously measure pH using an electrochemical **pH probe**.

## The problem

The pH probe is also the system's most fragile and maintenance-intensive
component. Electrochemical electrodes **degrade over time** through continuous
immersion, ion exchange, and electrode polarization; they suffer from
**calibration drift**, require frequent re-calibration, and have a limited
service life (often only a few months of continuous operation). This creates a
practical dilemma: reliable hydroponic automation depends on constant pH
measurement, yet constant measurement is exactly what wears the probe out
fastest. Existing IoT-based hydroponic systems largely treat the pH reading as
always-available and always-correct, offering little tolerance to sensor faults
and no mechanism to reduce the probe's workload.

## The proposed idea: a virtual sensor

This project addresses that dilemma by developing a **virtual pH sensor** — a
deep-learning model that *predicts* the pH of the nutrient solution from the
other, cheaper and more durable sensors in the system (total dissolved solids,
water temperature, air temperature, humidity, and water level). If these
predictions are accurate and trustworthy, the physical probe can be consulted
far less frequently, dramatically reducing its wear and extending its usable
lifespan, while the predicted value serves the control loop in between.
Technically, the task is framed as **one-step-ahead multivariate time-series
forecasting**: the model observes a ten-minute window of all sensor readings and
predicts the pH of the next minute.

The work is built upon, and benchmarked against, the research paper *"Toward
sustainable hydroponic farming: An AI-driven IoT framework for virtual pH
sensing and sensor lifespan extension"* (Moniruzzaman et al., *Alexandria
Engineering Journal*, 2026), and uses its publicly released Kaggle dataset of
real hydroponic IoT sensor and actuator logs (approximately 50,570 ten-second
readings over one month). The paper's central model is a **Gated Recurrent Unit
(GRU)**, a recurrent neural network well suited to sequential data, which it
selects for its strong balance of predictive accuracy and low computational cost
— a property essential for deployment on resource-constrained edge devices such
as a Raspberry Pi.

## Scope and contributions of this project

This project does three things, progressing from faithful reproduction to
original contribution:

**1. Faithful reproduction.** The complete machine-learning pipeline from the
paper is reconstructed from scratch — data cleaning (interpolation of
sensor-failure values, physical-range validation, and resampling),
normalization, windowing, and training of the GRU together with LSTM and
Transformer baselines. The reproduction matches the paper closely, reaching a
coefficient of determination of R² ≈ 0.98 and reproducing the GRU's exact
24,351-parameter architecture.

**2. A critical methodological finding.** On auditing the reproduced result,
this project identifies that the headline accuracy is inflated by **temporal
data leakage**: because overlapping time windows are shuffled randomly into the
training and test sets, the test data closely resembles the training data. The
project demonstrates this rigorously — a trivial "repeat-the-last-value"
persistence baseline also reaches R² ≈ 0.98 under the same protocol, proving the
score reflects memorization rather than genuine forecasting skill. A
**leakage-free temporal evaluation** (training on an earlier period and testing
on a strictly later period with an embargo gap) is introduced, under which the
original GRU's performance collapses — exposing a limitation that applies to the
broader literature, not just this reproduction.

**3. An honest redesign and a novel algorithm.** The GRU is redesigned using a
**residual (delta) formulation** and robust training (Huber loss, weight decay,
learning-rate scheduling, and early stopping), recovering strong, genuine
generalization at less than half the original parameter count. Building on this,
the project proposes **CARD (Confidence-Adaptive Residual Dosing-aware
forecaster)**, a novel fused algorithm combining a dosing-aware adaptive skip
connection, a tolerance-weighted loss, a self-supervised uncertainty estimate,
and a controller that targets a chosen physical-sensor usage budget. The
resulting system is further validated for **fault robustness** (remaining
accurate when the physical probe is corrupted) and **edge readiness**
(sub-millisecond CPU inference and a compact ONNX export).

## Guiding principle

Throughout, the project adheres to a principle of **measured honesty**: every
reported number is produced by a reproducible experiment, baselines are strong
rather than convenient, and claims are framed precisely — the system's value is
shown to lie not in an inflated accuracy figure, but in trustworthy,
confidence-aware virtual sensing that reduces physical-sensor usage, tolerates
sensor failure, and runs efficiently at the edge.
