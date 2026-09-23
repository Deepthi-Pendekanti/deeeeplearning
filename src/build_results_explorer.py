"""
Build a single self-contained Results Explorer dashboard (one HTML file).

It reads the REAL metric JSONs, embeds them + explanations, copies the relevant
figures as base64 (so the file works by double-click, no server needed), and
writes results/results_explorer.html.

Left sidebar = clickable list of every result section.
Right panel   = full detail for the selected result: what it is, the numbers,
                what it means, and the charts.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
REPORT_IMG = ROOT / "results" / "report_images"
OUT = ROOT / "results" / "results_explorer.html"


def load(name):
    p = METRICS / name
    return json.load(open(p)) if p.exists() else None


def img_b64(path: Path):
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def metric_table(rows, headers):
    """rows = list of dict; headers = list of (key, label)."""
    th = "".join(f"<th>{html.escape(l)}</th>" for _, l in headers)
    trs = []
    for r in rows:
        highlight = r.get("_hl")
        cls = ' class="hl"' if highlight else ""
        tds = "".join(f"<td>{html.escape(str(r.get(k,'')))}</td>"
                      for k, _ in headers)
        trs.append(f"<tr{cls}>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def explain(title, body_html):
    """A collapsible, richly-styled detailed-explanation block."""
    return (f'<div class="explain"><div class="ex-h">📘 {html.escape(title)}</div>'
            f'<div class="ex-b">{body_html}</div></div>')


def bullets(items):
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


# ---------------------------------------------------------------------------
# Build each section as {id, title, group, html}
# ---------------------------------------------------------------------------
def build_sections():
    S = []

    # ---- Overview ----
    S.append({
        "id": "overview", "group": "Start here", "title": "Project Overview",
        "html": f"""
        <h1>Virtual pH Sensor — Results Explorer</h1>
        <p class="lead">A GRU-based virtual pH sensor for hydroponic farming.
        Predict water pH from cheaper sensors so the fragile physical pH probe
        is used less and lasts longer. Based on Moniruzzaman et al.,
        Alexandria Engineering Journal 143 (2026).</p>
        <div class="cards">
          <div class="c"><div class="k">0.9824</div><div class="l">Paper R² (reproduced exactly)</div></div>
          <div class="c red"><div class="k">0.03</div><div class="l">Original GRU R² on honest test (fails)</div></div>
          <div class="c green"><div class="k">0.65</div><div class="l">Improved GRU R² on honest test</div></div>
          <div class="c violet"><div class="k">75.6%</div><div class="l">Physical sensor usage cut</div></div>
        </div>
        <p><b>The story:</b> we reproduced the paper (R²=0.98), discovered that
        number was inflated by <b>data leakage</b>, built an honest leakage-free
        test, fixed the GRU so it genuinely generalizes, and added a
        confidence system + fault-robustness that give the virtual sensor real
        value. Click any item on the left to see full details.</p>
        {img_tag("00_summary_card.png", REPORT_IMG)}
        {explain("What problem are we solving? (plain English)", bullets([
          "Hydroponics = growing plants in water instead of soil. The water must "
          "stay at the right <b>pH</b> (acidity) or plants suffer.",
          "pH is measured with a <b>physical probe</b> dipped in the water. These "
          "probes are fragile: they wear out, drift, and need frequent replacing.",
          "Our idea (a <b>virtual sensor</b>): train a model to <i>predict</i> pH "
          "from the other, cheaper, durable sensors (temperature, humidity, "
          "nutrient level, water level).",
          "If the prediction is trustworthy, we can read the real probe far less "
          "often → it lasts much longer and costs less to maintain.",
          "Technically this is <b>time-series forecasting</b>: use the last 10 "
          "minutes of sensor readings to predict the next minute's pH.",
        ]))}
        {explain("How to read the scores (important for the review)", bullets([
          "<b>R²</b> (R-squared, 0→1): the fraction of the pattern the model "
          "explains. 1 = perfect. It is <b>NOT</b> an accuracy percentage — never "
          "say 'R²=0.98 means 98% accurate'.",
          "<b>MAE</b> (Mean Absolute Error): the average size of the mistake, in "
          "pH units. Lower = better. 0.02 means we're off by ~0.02 pH on average.",
          "<b>RMSE</b>: like MAE but punishes big mistakes much harder. If RMSE is "
          "much bigger than MAE, a few large errors are dominating.",
          "<b>within ±0.05 pH</b>: the % of predictions within 0.05 of the truth — "
          "this is the closest thing to an intuitive 'accuracy %'.",
          "<b>Data leakage</b>: accidentally letting the model see answers it "
          "shouldn't during training. It makes scores look great but fake. This is "
          "the central finding of our project.",
        ]))}
        """,
    })

    # ---- Concepts glossary ----
    S.append({
        "id": "concepts", "group": "Start here",
        "title": "Concepts & glossary (for beginners)",
        "html": f"""
        <h1>Concepts & glossary</h1>
        <p class="lead">Every term used in this project, explained simply.</p>
        {explain("The models we used (10 total)", bullets([
          "<b>Persistence</b> — the 'dumb' baseline: predict next pH = last pH. "
          "Every real model must beat this to be worth anything.",
          "<b>Linear Regression</b> — fits a straight-line relationship. Simplest.",
          "<b>Random Forest / ExtraTrees</b> — many decision trees averaged; "
          "strong, robust tree models.",
          "<b>XGBoost</b> — gradient-boosted trees; a very popular, powerful ML model.",
          "<b>ARIMA</b> — a classic statistics forecasting method (uses only past pH).",
          "<b>GRU / LSTM</b> — 'recurrent' neural networks built for sequences; "
          "they remember recent history. GRU is the paper's main model.",
          "<b>Transformer</b> — an attention-based model (the tech behind modern AI).",
          "<b>Improved GRU</b> — our upgraded GRU (the 'delta head' version).",
        ]))}
        {explain("Key techniques", bullets([
          "<b>Look-back window = 10</b>: each prediction uses the previous 10 "
          "one-minute readings of all sensors.",
          "<b>Normalization (MinMax)</b>: squash every sensor to a 0–1 range so "
          "training is stable. We fit it on training data only (no leakage).",
          "<b>Train / Validation / Test split</b>: train = learn; validation = "
          "tune settings; test = the final unseen exam.",
          "<b>Random split</b> (leaky) vs <b>Temporal split</b> (honest): random "
          "shuffles windows so test overlaps train (cheating); temporal keeps a "
          "time gap so the test is a genuine future period.",
          "<b>Delta head</b>: instead of predicting absolute pH, predict the "
          "<i>change</i> and add it to the last value. Stops the model from "
          "cheating by copying the last reading.",
          "<b>Huber loss</b>: a training objective that ignores extreme outliers, "
          "so a couple of sensor glitches don't dominate.",
          "<b>MC-Dropout</b>: run the model several times with a bit of randomness; "
          "if predictions wobble a lot, the model is uncertain.",
          "<b>ONNX</b>: a portable model file format that runs on tiny edge devices "
          "like a Raspberry Pi.",
        ]))}
        """,
    })

    # ---- Honest comparison (Phase 4 temporal) ----
    d = load("phase4_baselines_paper_temporal.json")
    imp = load("phase5_search.json")
    rows = []
    order = ["Persistence", "LinearRegression", "RandomForest", "ExtraTrees",
             "XGBoost", "LSTM", "GRU"]
    for k in order:
        if d and k in d:
            v = d[k]
            rows.append({"Model": k, "R²": fmt(v["R2"], 3),
                         "MAE (pH)": fmt(v["MAE"]), "RMSE (pH)": fmt(v["RMSE"]),
                         "within ±0.05 pH": fmt(v["within_0.05pH"], 1) + "%"})
    if imp:
        t = imp["test_metrics"]
        rows.append({"Model": "Improved GRU (ours)", "R²": fmt(t["R2"], 3),
                     "MAE (pH)": fmt(t["MAE"]), "RMSE (pH)": fmt(t["RMSE"]),
                     "within ±0.05 pH": fmt(t["within_0.05pH"], 1) + "%",
                     "_hl": True})
    S.append({
        "id": "honest", "group": "Key results",
        "title": "★ Honest model comparison (leakage-free)",
        "html": f"""
        <h1>Model results — leakage-free (honest) test</h1>
        <p class="lead">Temporal split with an embargo gap so test data never
        overlaps training data. <b>This is the trustworthy comparison.</b></p>
        {metric_table(rows, [("Model","Model"),("R²","R²"),("MAE (pH)","MAE (pH)"),
          ("RMSE (pH)","RMSE (pH)"),("within ±0.05 pH","within ±0.05 pH")])}
        <div class="note"><b>What it means:</b> the original GRU fails here
        (R²=0.03) — worse than the trivial "repeat last value" persistence
        baseline. Our Improved GRU recovers to R²≈0.65, matching the strongest
        models (ExtraTrees, persistence) while using half the parameters.
        R² is "fraction of variation explained" (1=perfect), NOT an accuracy %.</div>
        {explain("Why this table is the one that matters", bullets([
          "This uses the <b>temporal (leakage-free) split</b>: we train on the "
          "earlier period, leave a gap, and test on a genuinely <i>future</i> "
          "period the model has never seen.",
          "That's how the sensor would really be used — predicting the future, "
          "not filling gaps between known points.",
          "Because there's no leakage, these numbers are lower than the paper's "
          "0.98 — but they are <b>honest</b>. Lower-but-honest beats "
          "higher-but-fake in any serious review.",
        ]))}
        {explain("How to read each row", bullets([
          "<b>Persistence (R²=0.65)</b> — the bar every model must clear. On this "
          "smooth 1-minute data, 'repeat the last value' is already strong "
          "because pH changes slowly.",
          "<b>Original GRU (R²=0.03)</b> — collapses. It had memorised the leaky "
          "signal and can't forecast a real future window.",
          "<b>LSTM (0.17), Linear (0.19), RandomForest (0.26)</b> — all weak here.",
          "<b>XGBoost (0.55), ExtraTrees (0.68)</b> — the tree models generalise "
          "best; ExtraTrees is the single best model on the honest test.",
          "<b>Improved GRU (0.65)</b> — our fix brings the GRU back up to match "
          "persistence and approach ExtraTrees, at less than half the parameters.",
        ]))}
        {explain("The honest takeaway to say out loud", 
          "<p>On clean data our Improved GRU <b>ties</b> the simple baselines "
          "rather than crushing them — because on a slow-moving signal, that's "
          "near the achievable ceiling. We say this openly. The GRU's real "
          "advantage is not this number; it's the <b>confidence-based sensor "
          "switching</b> and <b>fault robustness</b> (see the Novelty sections), "
          "which the simple baselines cannot provide.</p>")}
        {img_tag("model_comparison_temporal.png", FIGURES)}
        {img_tag("01_honest_comparison.png", REPORT_IMG)}
        """,
    })

    # ---- Paper reproduction (Phase 4 random) ----
    d = load("phase4_baselines_paper_random.json")
    rows = []
    for k in ["GRU", "LSTM", "Persistence", "ExtraTrees", "XGBoost"]:
        if d and k in d:
            v = d[k]
            rows.append({"Model": k, "R²": fmt(v["R2"]),
                         "MAE (pH)": fmt(v["MAE"]), "RMSE (pH)": fmt(v["RMSE"]),
                         "within ±0.05 pH": fmt(v["within_0.05pH"], 1) + "%",
                         "_hl": k == "GRU"})
    S.append({
        "id": "repro", "group": "Key results",
        "title": "Paper reproduction (random split)",
        "html": f"""
        <h1>Paper reproduction — random split</h1>
        <p class="lead">We reproduced the paper's R²=0.98 regime exactly.</p>
        {metric_table(rows, [("Model","Model"),("R²","R²"),("MAE (pH)","MAE (pH)"),
          ("RMSE (pH)","RMSE (pH)"),("within ±0.05 pH","within ±0.05 pH")])}
        <div class="warn"><b>⚠ Important:</b> even the trivial <b>Persistence</b>
        predictor reaches R²=0.98 here. That is the tell-tale sign of
        <b>temporal data leakage</b> — the score is inflated, not real skill.
        This is why we built the honest test.</div>
        {explain("What exactly is the leakage here?", bullets([
          "The 'random split' first cuts the data into 10-minute windows, then "
          "<b>shuffles the windows</b> into train/test.",
          "Two neighbouring windows overlap by 9 of their 10 minutes and have "
          "almost-identical answers.",
          "So a test window can be nearly the same as one the model trained on — "
          "the model has effectively already seen the answer.",
          "Result: every model, even the dumb persistence one, scores ~0.98. The "
          "score measures memorisation, not forecasting ability.",
        ]))}
        {explain("Did we do something wrong? No — we reproduced the paper faithfully",
          "<p>We first matched the paper's exact setup and got R²=0.9824 — proving "
          "our reproduction is correct. Then, auditing it, we <b>found</b> the "
          "leakage that this setup contains. Exposing that is a contribution, not "
          "a mistake. We keep this table only to show 'yes, we can reproduce the "
          "published number', then we move to the honest evaluation.</p>")}
        {img_tag("02_paper_reproduction.png", REPORT_IMG)}
        {img_tag("fig16_actual_vs_predicted.png", FIGURES)}
        """,
    })

    # ---- Improvement before/after ----
    base = load("phase2_baseline_paper_temporal.json")
    b = base["gru_baseline"] if base else {}
    t = imp["test_metrics"] if imp else {}
    rows = [
        {"Metric": "R²", "Original GRU": fmt(b.get("R2",0),3),
         "Improved GRU": fmt(t.get("R2",0),3), "Change": "+0.62", "_hl": True},
        {"Metric": "MAE (pH)", "Original GRU": fmt(b.get("MAE",0)),
         "Improved GRU": fmt(t.get("MAE",0)), "Change": "−50%"},
        {"Metric": "RMSE (pH)", "Original GRU": fmt(b.get("RMSE",0)),
         "Improved GRU": fmt(t.get("RMSE",0)), "Change": "−40%"},
        {"Metric": "within ±0.05 pH", "Original GRU": fmt(b.get("within_0.05pH",0),1)+"%",
         "Improved GRU": fmt(t.get("within_0.05pH",0),1)+"%", "Change": "+22 pts"},
        {"Metric": "Parameters", "Original GRU": "24,051",
         "Improved GRU": "10,733", "Change": "2.24× smaller"},
    ]
    S.append({
        "id": "improvement", "group": "Key results",
        "title": "Improvement: original → improved GRU",
        "html": f"""
        <h1>Improvement — original vs improved GRU</h1>
        <p class="lead">Same honest leakage-free test for both models.</p>
        {metric_table(rows, [("Metric","Metric"),("Original GRU","Original GRU"),
          ("Improved GRU","Improved GRU"),("Change","Change")])}
        <div class="note"><b>What changed:</b> the improved GRU predicts the
        <b>change</b> in pH (delta head) instead of the absolute value, plus
        Huber loss, dropout, weight decay, LR scheduler and early stopping.
        The delta head is the single change responsible for the jump.</div>
        {explain("Why the 'delta head' is the key fix", bullets([
          "The old GRU predicted the <b>absolute pH</b>. The easiest way to score "
          "well under leakage is to just echo the last value — so it never "
          "learned real structure and collapsed on the honest test.",
          "The new GRU predicts the <b>change</b> (Δ) from the last pH, then adds "
          "it back: <i>next pH = last pH + predicted change</i>.",
          "Now the network's only job is to learn the small <i>deviation</i> from "
          "'stay the same' — which is exactly the useful signal.",
          "Our ablation (Phase 5) proved it: every config <b>with</b> the delta "
          "head scored R²≈0.71 on validation; every config <b>without</b> it was "
          "near-zero or negative.",
        ]))}
        {explain("The supporting techniques (each does one job)", bullets([
          "<b>Huber loss</b> — stops 2 glitchy pH spikes from dominating training.",
          "<b>Dropout + weight decay</b> — prevent over-fitting (memorising).",
          "<b>Learning-rate scheduler</b> — automatically slows learning as it "
          "converges, for a cleaner final model.",
          "<b>Early stopping</b> — stop when validation stops improving, keep the "
          "best model (not the last one).",
          "<b>Result:</b> better accuracy AND a smaller model (10,733 vs 24,051 "
          "parameters) — 2.24× fewer knobs.",
        ]))}
        {explain("How to read the loss curve below",
          "<p>The chart shows training loss and validation loss dropping over "
          "epochs. They fall together and flatten out — a sign of healthy "
          "learning with no over-fitting. Early stopping halts training once the "
          "validation loss stops improving.</p>")}
        {img_tag("03_improvement.png", REPORT_IMG)}
        {img_tag("improved_loss_curve.png", FIGURES)}
        """,
    })

    # ---- Confidence / sensor usage (Phase 7) ----
    c = load("phase7_confidence.json")
    if c:
        hm = c["hybrid_metrics"]; vm = c["pure_virtual_metrics"]
        rows = [
            {"System": "Always physical (traditional)", "Physical reads": "100%",
             "R²": "exact", "MAE (pH)": "0"},
            {"System": "Pure virtual (no switching)", "Physical reads": "0%",
             "R²": fmt(vm["R2"],3), "MAE (pH)": fmt(vm["MAE"])},
            {"System": "Hybrid (confidence switching)",
             "Physical reads": fmt(100*c["physical_fraction"],1)+"%",
             "R²": fmt(hm["R2"],3), "MAE (pH)": fmt(hm["MAE"]), "_hl": True},
        ]
        S.append({
            "id": "confidence", "group": "Novelty",
            "title": "★ Confidence-aware sensor switching",
            "html": f"""
            <h1>Confidence-aware hybrid sensing</h1>
            <p class="lead">The model estimates its own uncertainty
            (MC-Dropout) and only calls the physical probe when unsure.</p>
            {metric_table(rows, [("System","System"),("Physical reads","Physical reads"),
              ("R²","R²"),("MAE (pH)","MAE (pH)")])}
            <div class="big-stat">{fmt(c['reduction_pct'],1)}%
              <span>fewer physical-probe reads</span></div>
            <div class="note"><b>What it means:</b> the hybrid system cut
            physical sensor usage by {fmt(c['reduction_pct'],1)}% while keeping
            accuracy — this is exactly the "extend sensor lifespan" goal, and
            it's <b>measured</b>, not assumed.</div>
            {explain("How the model 'knows when it's unsure' (MC-Dropout)", bullets([
              "Dropout normally randomly switches off parts of the network during "
              "training. We keep it ON at prediction time and run the model ~30 "
              "times on the same input.",
              "If those 30 predictions <b>agree</b> → the model is confident. If "
              "they <b>scatter</b> → the model is unsure. The spread (std) is our "
              "uncertainty score.",
              "The 'uncertainty vs error' chart below confirms this works: higher "
              "uncertainty really does line up with larger prediction errors.",
            ]))}
            {explain("The switching policy (how it saves the probe)", bullets([
              "Start in <b>virtual mode</b> — trust the model's prediction.",
              "If uncertainty rises above a threshold, OR the tracked error drifts "
              "too high → switch to <b>physical mode</b> and read the real probe.",
              "Thresholds were tuned on <b>validation</b> data (not the test), so "
              "the reported reduction is a fair estimate.",
              "A slow 'safety cadence' still reads the probe occasionally even "
              "when confident, as a sanity check.",
              f"<b>Net result:</b> the probe is read only {fmt(100*c['physical_fraction'],1)}% "
              "of the time instead of 100% — a "
              f"{fmt(c['reduction_pct'],1)}% reduction.",
            ]))}
            {explain("Honest wording for the review",
              "<p>We report the <b>measured reduction in probe reads</b> "
              "(~76%). We do <b>not</b> claim a specific 'lifespan × N' number, "
              "because that needs a probe-wear model we didn't build. But since "
              "electrode wear grows with usage, far fewer reads is expected to "
              "meaningfully extend service life.</p>")}
            {img_tag("sensor_usage.png", FIGURES)}
            {img_tag("uncertainty_vs_error.png", FIGURES)}
            """,
        })

    # ---- Fault robustness (Phase 8) ----
    f = load("phase8_failure.json")
    if f:
        to = f["traditional_overall"]; po = f["proposed_overall"]
        tf = f["traditional_during_fault"]; pf = f["proposed_during_fault"]
        rows = [
            {"System": "Traditional (overall)", "R²": fmt(to["R2"],2),
             "MAE (pH)": fmt(to["MAE"]), "RMSE (pH)": fmt(to["RMSE"]),
             "within ±0.05": fmt(to["within_0.05pH"],1)+"%"},
            {"System": "Proposed (overall)", "R²": fmt(po["R2"],2),
             "MAE (pH)": fmt(po["MAE"]), "RMSE (pH)": fmt(po["RMSE"]),
             "within ±0.05": fmt(po["within_0.05pH"],1)+"%", "_hl": True},
            {"System": "Traditional (during fault)", "R²": fmt(tf["R2"],2),
             "MAE (pH)": fmt(tf["MAE"]), "RMSE (pH)": fmt(tf["RMSE"]),
             "within ±0.05": fmt(tf["within_0.05pH"],1)+"%"},
            {"System": "Proposed (during fault)", "R²": fmt(pf["R2"],2),
             "MAE (pH)": fmt(pf["MAE"]), "RMSE (pH)": fmt(pf["RMSE"]),
             "within ±0.05": fmt(pf["within_0.05pH"],1)+"%", "_hl": True},
        ]
        S.append({
            "id": "faults", "group": "Novelty",
            "title": "★ Sensor-failure robustness",
            "html": f"""
            <h1>Robustness when the pH sensor fails</h1>
            <p class="lead">30% of the stream corrupted (missing, frozen,
            noisy, drifting, miscalibrated).</p>
            {metric_table(rows, [("System","System"),("R²","R²"),("MAE (pH)","MAE (pH)"),
              ("RMSE (pH)","RMSE (pH)"),("within ±0.05","within ±0.05")])}
            <div class="note"><b>What it means:</b> a traditional system that
            trusts the broken probe collapses (R²=−6.9). The proposed system
            detects the fault and falls back to the virtual sensor, staying
            accurate (R²=0.89). This is a benefit persistence cannot provide.</div>
            {explain("What faults did we simulate, and why?", bullets([
              "<b>Missing</b> — the probe returns nothing (disconnected).",
              "<b>Dropout/frozen</b> — the reading gets stuck at one value.",
              "<b>Noise</b> — random jitter added to the reading.",
              "<b>Drift</b> — a slowly growing bias (electrode ageing).",
              "<b>Calibration error</b> — a constant offset (badly calibrated).",
              "These are the five most common real ways a pH probe goes bad. We "
              "corrupt 30% of the test stream with them.",
            ]))}
            {explain("Why the traditional system gets a NEGATIVE R²",
              "<p>R² can go below zero when a model is <i>worse than just guessing "
              "the average</i>. The traditional system blindly trusts the "
              "corrupted probe, so during faults it outputs badly wrong values — "
              "worse than a flat line. Our system detects the fault and switches "
              "to the virtual prediction, so it stays close to the truth "
              "(R²=0.89 overall, 0.71 even during the fault windows).</p>")}
            {explain("Why this is the strongest argument for the project",
              "<p>On clean data, a simple predictor matches our model. But a "
              "simple predictor <b>has no idea when the sensor is lying</b> — it "
              "will happily pass on a frozen or miscalibrated reading. Our "
              "confidence + fallback design is what makes it a real, deployable "
              "safety feature. The chart below shows the green (proposed) line "
              "staying on the true pH while the red (traditional) line diverges "
              "in every shaded fault window.</p>")}
            {img_tag("failure_sim.png", FIGURES)}
            """,
        })

    # ---- Efficiency (Phase 10) ----
    e = load("phase10_efficiency.json")
    if e and "ImprovedGRU" in e:
        g = e.get("GRU_baseline", {}); im = e["ImprovedGRU"]
        rows = [
            {"Model": "GRU baseline", "Params": f"{g.get('params',0):,}",
             "Size (.pt)": fmt(g.get("disk_kb",0),1)+" KB",
             "Latency (mean)": fmt(g.get("latency",{}).get("mean_ms",0),3)+" ms"},
            {"Model": "Improved GRU", "Params": f"{im.get('params',0):,}",
             "Size (.pt)": fmt(im.get("disk_kb",0),1)+" KB",
             "Latency (mean)": fmt(im.get("latency",{}).get("mean_ms",0),3)+" ms",
             "_hl": True},
        ]
        onnx = im.get("onnx", {})
        onnx_txt = (f"exported {fmt(onnx.get('size_kb',0),1)} KB, parity "
                    f"{onnx.get('max_abs_diff','?')}" if onnx.get("exported")
                    else "not exported")
        S.append({
            "id": "efficiency", "group": "Deployment",
            "title": "Model efficiency (edge-ready)",
            "html": f"""
            <h1>Model efficiency & edge readiness</h1>
            {metric_table(rows, [("Model","Model"),("Params","Params"),
              ("Size (.pt)","Size (.pt)"),("Latency (mean)","Latency (mean)")])}
            <div class="note"><b>ONNX export (Improved GRU):</b> {onnx_txt}.
            Small and fast enough for a Raspberry Pi — sub-millisecond CPU
            inference and a ~7 KB portable model file.</div>
            {explain("Why efficiency matters for this project", bullets([
              "The whole point is to run <b>on the edge</b> — on a cheap device "
              "sitting by the hydroponic tank, not in the cloud.",
              "<b>Latency</b> = how long one prediction takes. Ours is ~0.9 "
              "milliseconds on a plain CPU — effectively instant.",
              "<b>Parameters / size</b> = how much memory it needs. 10,733 "
              "parameters and a ~7 KB ONNX file fit trivially on a Raspberry Pi.",
              "<b>ONNX</b> is a universal model format. 'Parity 0.0' means the "
              "exported ONNX model gives <i>identical</i> outputs to the original "
              "PyTorch model — so nothing breaks when deployed.",
              "Bonus: the improved model is <b>smaller</b> than the baseline "
              "(10,733 vs 24,051 params) yet more accurate on the honest test.",
            ]))}
            """,
        })

    # ---- Error analysis (Phase 3) ----
    S.append({
        "id": "errors", "group": "Analysis",
        "title": "Error analysis (why RMSE > MAE)",
        "html": f"""
        <h1>Largest-error analysis</h1>
        <p class="lead">Why is RMSE bigger than MAE?</p>
        <div class="note">A single worst prediction contributed ~49% of the
        total squared error, and the top-2 were physically-impossible one-step
        pH jumps (+0.83, +0.49) with no actuator active — uncaught sensor
        glitches. Removing just the worst 2 brings RMSE down to ≈0.017.
        We did <b>not</b> delete them; we used Huber loss to reduce their
        influence instead.</div>
        {explain("Why does RMSE look worse than MAE?", bullets([
          "<b>MAE</b> averages the mistakes normally. <b>RMSE</b> squares them "
          "first — so one big mistake counts enormously.",
          "We found the single largest error alone caused ~49% of the total "
          "squared error. A handful of points were inflating RMSE.",
          "Those points were <b>+0.83 and +0.49 pH jumps in a single minute</b> "
          "with no dosing happening — physically impossible, i.e. leftover "
          "sensor glitches, not model failures.",
          "So the model is actually accurate (median error ≈0.011 pH); RMSE was "
          "just being dragged up by 1–2 bad data points.",
        ]))}
        {explain("What we did about it (the honest, correct choice)",
          "<p>We did <b>not</b> quietly delete the awkward points to make numbers "
          "look better — that would be dishonest. Instead we switched to "
          "<b>Huber loss</b>, which naturally reduces the training influence of "
          "extreme outliers while keeping all the data. That's the principled "
          "fix.</p>")}
        {img_tag("largest_errors.png", FIGURES)}
        {img_tag("improved_residual_hist.png", FIGURES)}
        """,
    })

    # ---- All deep models (paper reproduction: GRU vs LSTM vs Transformer) ----
    dl = load("training_results_paper_6feat_random.json")
    if dl and "results" in dl:
        r = dl["results"]
        rows = []
        for m, label in [("gru", "GRU"), ("lstm", "LSTM"),
                         ("transformer", "Transformer")]:
            if m in r:
                b = r[m]["best_run"]
                rows.append({"Model": label, "R²": fmt(b["R2"], 4),
                             "MAE (pH)": fmt(b["MAE"]),
                             "RMSE (pH)": fmt(b["RMSE"]),
                             "Params": f"{r[m]['params']:,}",
                             "_hl": m == "gru"})
        S.append({
            "id": "deepmodels", "group": "All models",
            "title": "Deep models: GRU vs LSTM vs Transformer",
            "html": f"""
            <h1>Deep learning models — paper reproduction</h1>
            <p class="lead">The three neural networks from the paper, trained
            under identical settings (3 seeds, best run shown). Random split.</p>
            {metric_table(rows, [("Model","Model"),("R²","R²"),("MAE (pH)","MAE (pH)"),
              ("RMSE (pH)","RMSE (pH)"),("Params","Params")])}
            <div class="note"><b>What it means:</b> GRU and LSTM are essentially
            tied and both far more compact (~24k parameters) than the
            Transformer (~351k). The paper picks the GRU for its lower inference
            latency. Note: these are the random-split (leaky) numbers — see the
            honest comparison for the trustworthy ranking.</div>
            {explain("What are GRU, LSTM and Transformer?", bullets([
              "<b>GRU</b> and <b>LSTM</b> are 'recurrent' networks: they read the "
              "sequence one step at a time and carry a memory of what they've "
              "seen. Good for time-series. GRU is a simpler, faster LSTM.",
              "<b>Transformer</b> uses 'attention' to look at all 10 time steps at "
              "once and weigh their importance. Powerful but much heavier here "
              "(351k parameters vs 24k).",
              "<b>Parameters</b> = the model's tunable knobs. Fewer = smaller, "
              "faster, cheaper to run on small devices — which matters for a "
              "Raspberry-Pi deployment.",
            ]))}
            {explain("Why the paper (and we) pick GRU",
              "<p>GRU and LSTM reach basically the same accuracy, but the GRU is "
              "compact and has lower inference latency, making it the best fit "
              "for a low-power edge device. The Transformer's extra size buys no "
              "accuracy here, so it's not worth it. The charts below compare all "
              "three: predictions vs truth, error distributions, and "
              "outlier counts.</p>")}
            {img_tag("fig16_actual_vs_predicted.png", FIGURES)}
            {img_tag("fig18_residual_histograms.png", FIGURES)}
            {img_tag("fig19_threshold_outliers.png", FIGURES)}
            """,
        })

    # ---- ARIMA classical baseline ----
    bl = load("baselines.json")
    if bl and bl.get("arima"):
        a = bl["arima"]; pa = bl.get("paper_arima", {})
        rows = [
            {"Version": "Our ARIMA(3,1,3)", "R²": fmt(a["R2"], 3),
             "MAE (pH)": fmt(a["MAE"]), "RMSE (pH)": fmt(a["RMSE"]),
             "AIC": fmt(a.get("aic", 0), 1), "_hl": True},
        ]
        if pa:
            rows.append({"Version": "Paper ARIMA(3,1,3)",
                         "R²": str(pa.get("R2", "?")),
                         "MAE (pH)": str(pa.get("MAE", "?")),
                         "RMSE (pH)": str(pa.get("RMSE", "?")),
                         "AIC": str(pa.get("AIC", "?"))})
        S.append({
            "id": "arima", "group": "All models",
            "title": "ARIMA (classical statistics baseline)",
            "html": f"""
            <h1>ARIMA — classical statistical baseline</h1>
            <p class="lead">A traditional time-series model using only past pH.
            We grid-searched the (p,d,q) order by AIC and it selected
            <b>(3,1,3)</b> — the same order the paper reports.</p>
            {metric_table(rows, [("Version","Version"),("R²","R²"),("MAE (pH)","MAE (pH)"),
              ("RMSE (pH)","RMSE (pH)"),("AIC","AIC")])}
            <div class="note"><b>What it means:</b> ARIMA only looks at past pH
            and cannot use the other sensors, so it is much weaker than the deep
            and tree models. This is exactly why a multivariate model is needed.
            (The paper reported a negative R² on their test window; on the
            smoother subset ours reaches R²≈0.72 — but still clearly the weakest
            approach.)</div>
            {explain("What is ARIMA and what do (p,d,q) mean?", bullets([
              "<b>ARIMA</b> = AutoRegressive Integrated Moving Average — a classic "
              "statistical forecasting method from before deep learning.",
              "<b>p</b> = how many past values it uses; <b>d</b> = how many times "
              "it differences the series to remove trends; <b>q</b> = how many "
              "past errors it corrects for.",
              "We tried many (p,d,q) combinations and picked the best by <b>AIC</b> "
              "(a score balancing fit vs complexity). It chose (3,1,3) — the same "
              "as the paper, confirming our reproduction.",
              "Its big limitation: it only sees <b>past pH</b>, not temperature, "
              "nutrients, etc. Since pH depends on those, ARIMA is fundamentally "
              "limited here — which motivates the multivariate GRU/tree models.",
            ]))}
            {img_tag("fig9_correlation_matrix.png", FIGURES)}
            """,
        })

    # ---- Classical ML baselines (Table 6) ----
    if bl and bl.get("ml"):
        ml = bl["ml"]
        rows = []
        for name in ["MLP", "KNN", "XGBoost"]:
            if name in ml:
                v = ml[name]
                rows.append({"Model": name, "R²": fmt(v["R2"], 4),
                             "MAE (pH)": fmt(v["MAE"]),
                             "RMSE (pH)": fmt(v["RMSE"]),
                             "MAPE": fmt(v.get("MAPE", 0), 3) + "%",
                             "latency (ms)": fmt(v.get("latency_s", 0) * 1000, 3)})
        S.append({
            "id": "mlbaselines", "group": "All models",
            "title": "Classical ML: MLP / KNN / XGBoost",
            "html": f"""
            <h1>Classical machine-learning baselines</h1>
            <p class="lead">Non-deep learning models on the windowed features
            (the paper's Table 6 comparison).</p>
            {metric_table(rows, [("Model","Model"),("R²","R²"),("MAE (pH)","MAE (pH)"),
              ("RMSE (pH)","RMSE (pH)"),("MAPE","MAPE"),("latency (ms)","latency (ms)")])}
            <div class="note"><b>What it means:</b> tree-based XGBoost is a very
            strong, fast baseline — a key reason we compared against it. On the
            honest leakage-free test, tree models (ExtraTrees/XGBoost) were among
            the best, which is why our final claim is honest rather than
            "the GRU beats everything."</div>
            {explain("What are these three models?", bullets([
              "<b>MLP</b> (Multi-Layer Perceptron) — a basic neural network with "
              "fully-connected layers; no memory of sequence order.",
              "<b>KNN</b> (K-Nearest Neighbours) — predicts by finding the most "
              "similar past situations and averaging their pH.",
              "<b>XGBoost</b> — builds many small decision trees in sequence, each "
              "fixing the previous ones' mistakes; extremely strong on tabular "
              "data and very fast.",
              "We feed them the flattened 10-minute window as input features.",
            ]))}
            {explain("Why include them at all?",
              "<p>Good science compares a new model against strong, simple "
              "alternatives — not just weak ones. By testing XGBoost and the tree "
              "models, we can honestly answer 'is the GRU actually the best "
              "choice?' The answer on the honest split was nuanced (trees do "
              "very well too), which is why our final report is careful and "
              "truthful rather than over-claiming.</p>")}
            """,
        })

    # ---- Feature correlation ----
    S.append({
        "id": "features", "group": "Analysis",
        "title": "Feature correlation (Fig. 9)",
        "html": f"""
        <h1>Which sensors relate to pH?</h1>
        <p class="lead">Pearson correlation of every sensor with pH.</p>
        <div class="note">Reproduces the paper's Figure 9. The exhaust fan has
        ≈0 correlation with pH; pH depends on several interdependent variables,
        which is why a multivariate model is used instead of a univariate one.</div>
        {explain("What is a correlation heatmap?", bullets([
          "<b>Correlation</b> measures how strongly two things move together, "
          "from −1 (opposite) through 0 (unrelated) to +1 (move together).",
          "Each coloured square shows the correlation between two sensors. "
          "Read the 'pH' row/column to see what relates to pH.",
          "Sensors like temperature, TDS (nutrients) and humidity show real "
          "relationships with pH; the exhaust fan shows ≈0 (irrelevant).",
          "Takeaway: no single sensor determines pH — several matter together — "
          "so a model that combines them all (multivariate) is the right choice.",
        ]))}
        {img_tag("fig9_correlation_matrix.png", FIGURES)}
        """,
    })

    return S


def img_tag(name, folder):
    src = img_b64(folder / name)
    if not src:
        return ""
    return (f'<div class="fig"><img src="{src}" alt="{html.escape(name)}"/>'
            f'<div class="cap">{html.escape(name)}</div></div>')


def main():
    sections = build_sections()

    # group the nav
    groups = {}
    for s in sections:
        groups.setdefault(s["group"], []).append(s)

    nav = ""
    for g, items in groups.items():
        nav += f'<div class="navgroup">{html.escape(g)}</div>'
        for s in items:
            nav += (f'<a class="navitem" data-target="{s["id"]}" '
                    f'onclick="show(\'{s["id"]}\')">{html.escape(s["title"])}</a>')

    panels = ""
    for i, s in enumerate(sections):
        style = "" if i == 0 else "display:none"
        panels += f'<section id="{s["id"]}" class="panel" style="{style}">{s["html"]}</section>'

    first_id = sections[0]["id"]

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Virtual pH Sensor — Results Explorer</title>
<style>
*{{box-sizing:border-box;font-family:'Segoe UI',system-ui,Arial,sans-serif}}
body{{margin:0;background:#0e1420;color:#e8eefc}}
.layout{{display:flex;min-height:100vh}}
.side{{width:290px;background:#131c2e;border-right:1px solid #26324c;
  padding:18px 0;position:sticky;top:0;height:100vh;overflow:auto}}
.side h2{{font-size:14px;margin:0 20px 12px;color:#fff}}
.navgroup{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:#6f83a8;margin:16px 20px 6px}}
.navitem{{display:block;padding:9px 20px;color:#c4d2ee;font-size:14px;
  cursor:pointer;border-left:3px solid transparent}}
.navitem:hover{{background:#1b273e}}
.navitem.active{{background:#1b273e;border-left-color:#4aa3ff;color:#fff;font-weight:600}}
.main{{flex:1;padding:32px 44px;max-width:1000px}}
h1{{font-size:24px;margin:0 0 10px}}
.lead{{color:#9fb0d0;font-size:15px;margin:0 0 18px}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}}
th,td{{border:1px solid #2a3854;padding:9px 12px;text-align:center}}
th{{background:#22314f;color:#fff}}
td:first-child,th:first-child{{text-align:left}}
tr:nth-child(even) td{{background:#161f30}}
tr.hl td{{background:#123b2a;font-weight:700;color:#8ff0bf}}
.note{{background:#152238;border-left:4px solid #4aa3ff;padding:12px 16px;
  border-radius:6px;margin:16px 0;font-size:14px;line-height:1.6}}
.warn{{background:#2a1620;border-left:4px solid #ff5d6c;padding:12px 16px;
  border-radius:6px;margin:16px 0;font-size:14px;line-height:1.6}}
.fig{{margin:18px 0;background:#fff;border-radius:10px;padding:10px;text-align:center}}
.fig img{{max-width:100%;border-radius:6px}}
.cap{{color:#556;font-size:11px;margin-top:6px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}}
.c{{background:#17223a;border:1px solid #2a3854;border-radius:12px;padding:16px;text-align:center}}
.c .k{{font-size:26px;font-weight:800}}
.c .l{{font-size:11px;color:#9fb0d0;margin-top:6px}}
.c.red{{border-color:#ff5d6c}} .c.red .k{{color:#ff8a94}}
.c.green{{border-color:#26d07c}} .c.green .k{{color:#7bf0b4}}
.c.violet{{border-color:#a98bff}} .c.violet .k{{color:#c9b6ff}}
.big-stat{{font-size:40px;font-weight:800;color:#7bf0b4;margin:16px 0}}
.big-stat span{{display:block;font-size:14px;color:#9fb0d0;font-weight:400}}
.explain{{background:#101a2c;border:1px solid #2a3854;border-radius:10px;
  margin:16px 0;overflow:hidden}}
.ex-h{{background:#1a2740;padding:10px 16px;font-weight:700;font-size:14px;
  color:#cfe0ff}}
.ex-b{{padding:8px 18px 4px;font-size:14px;line-height:1.65;color:#cdd8ee}}
.ex-b ul{{margin:8px 0;padding-left:20px}}
.ex-b li{{margin:6px 0}}
.ex-b p{{margin:8px 0}}
.ex-b b{{color:#fff}}
.tophint{{color:#6f83a8;font-size:12px;padding:0 20px 10px}}
</style></head><body>
<div class="layout">
  <nav class="side">
    <h2>🌱 Results Explorer</h2>
    <div class="tophint">Click any result to see full details.</div>
    {nav}
  </nav>
  <main class="main">{panels}</main>
</div>
<script>
function show(id){{
  document.querySelectorAll('.panel').forEach(p=>p.style.display='none');
  document.getElementById(id).style.display='block';
  document.querySelectorAll('.navitem').forEach(a=>a.classList.remove('active'));
  document.querySelector('.navitem[data-target="'+id+'"]').classList.add('active');
  window.scrollTo(0,0);
}}
show('{first_id}');
</script>
</body></html>"""

    OUT.write_text(doc, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"[built] {OUT}  ({kb:.0f} KB, {len(sections)} sections, images embedded)")


if __name__ == "__main__":
    main()
