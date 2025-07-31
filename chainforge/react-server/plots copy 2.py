# plots.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64

import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

app = Flask(__name__)
CORS(app)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def fig_to_base64(fig: plt.Figure) -> str:
    """Serialize a Matplotlib figure to a base64‑encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def get_llm_name(resp: dict, llm_group: str) -> str:
    """Extract the LLM name or metavars group value."""
    if llm_group == "LLM":
        llm = resp.get("llm")
        return llm if isinstance(llm, str) else llm.get("name", "unknown")
    # strip "__meta_" prefix if present
    key = llm_group.replace("__meta_", "")
    return resp.get("metavars", {}).get(key, "unknown")


def get_y_value(resp: dict, selected_var: str) -> float:
    """
    Try resp['vars'][selected_var] first; if not numeric,
    fall back to averaging resp['eval_res']['items'].
    """
    raw = resp.get("vars", {}).get(selected_var)
    if raw is not None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass

    items = resp.get("eval_res", {}).get("items") or []
    nums = [v for v in items if isinstance(v, (int, float))]
    return float(np.mean(nums)) if nums else 0.0


def compute_classification_metrics(y_true: list, y_pred: list) -> dict:
    """Compute F1, FPR, FNR, and ROC‑AUC for binary labels."""
    y_t = [int(v) for v in y_true]
    y_p = [int(v) for v in y_pred]

    f1  = f1_score(y_t, y_p, average="weighted")
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    roc = roc_auc_score(y_t, y_p) if len(set(y_t)) > 1 else 0.0

    return {
        "F1 Score": f1,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr,
        "ROC AUC": roc,
    }


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/plot", methods=["POST"])
def plot():
    """
    General plotting endpoint.
    Expects JSON with:
      - selected_var: str
      - llm_group: str
      - plot_type: "bar" | "line" | "confusion_matrix"
      - responses: list of LLMResponse-like dicts
    Returns base64‑PNG under {"image": ...}.
    """
    payload      = request.json or {}
    sel_var      = payload.get("selected_var")
    llm_group    = payload.get("llm_group", "LLM")
    plot_type    = payload.get("plot_type", "bar")
    responses    = payload.get("responses", [])

    # Build x/y arrays
    x_vals = [get_llm_name(r, llm_group) for r in responses]
    y_vals = [get_y_value(r, sel_var)         for r in responses]

    fig, ax = plt.subplots()

    if plot_type == "line":
        ax.plot(x_vals, y_vals, marker="o")
    elif plot_type == "matrix":
        # Build binary true/pred from vars or eval_res
        y_true = [int(r["metavars"].get("true_label", 0))      for r in responses]
        y_pred = [int(r["metavars"].get("predicted_label", 0)) for r in responses]
        cm = confusion_matrix(y_true, y_pred)
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax, ax=ax)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="white")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
    else:  # "bar"
        ax.bar(x_vals, y_vals)

    ax.set_xlabel(llm_group)
    ax.set_ylabel(sel_var or "value")
    ax.set_title(f"{plot_type.title()} of {sel_var} by {llm_group}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return jsonify({"image": fig_to_base64(fig)})


@app.route("/metrics", methods=["POST"])
def metrics():
    """
    Compute raw classification metrics over all responses.
    Expects JSON with responses[], each having:
      - metavars.true_label
      - metavars.predicted_label OR eval_res.items[0]
    Returns JSON: { "F1 Score": ..., ... }.
    """
    payload   = request.json or {}
    responses = payload.get("responses", [])

    y_true = []
    y_pred = []
    for r in responses:
        tv = r.get("metavars", {}).get("true_label")
        pv = r.get("metavars", {}).get("predicted_label")
        if tv is None and "eval_res" in r:
            # fallback to first eval_res item
            items = r["eval_res"].get("items") or []
            pv    = items[0] if items else 0
        y_true.append(tv)
        y_pred.append(pv)

    results = compute_classification_metrics(y_true, y_pred)
    return jsonify(results)


@app.route("/metrics_plot", methods=["POST"])
def metrics_plot():
    """
    Bar‐chart of the metrics returned by /metrics.
    Returns a PNG under {"image": ...}.
    """
    m = metrics().get_json()
    fig, ax = plt.subplots()
    ax.bar(m.keys(), m.values(), color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return jsonify({"image": fig_to_base64(fig)})


@app.route("/metrics_heatmap", methods=["POST"])
def metrics_heatmap():
    """
    Compute per‐group metric (mean of eval_res.items[0]) and render as heatmap.
    Expects JSON with:
      - responses: list
      - group_by: metavars key name (e.g. "__meta_ids", "true_label", etc.)
      - metric: name to label the row
    Returns PNG under {"image": ...}.
    """
    payload   = request.json or {}
    responses = payload.get("responses", [])
    group_by  = payload.get("group_by", "__meta_ids").replace("__meta_", "")
    metric    = payload.get("metric", "Score")

    # Group responses
    groups = {}
    for r in responses:
        key = r.get("metavars", {}).get(group_by, "unknown")
        groups.setdefault(key, []).append(r)

    # Compute metric per group (mean of first eval_res item)
    data = {}
    for grp, items in groups.items():
        vals = []
        for r in items:
            ev = r.get("eval_res", {}).get("items") or []
            if ev:
                nums = [v for v in ev if isinstance(v, (int, float))]
                if nums:
                    vals.append(float(np.mean(nums)))
        data[grp] = float(np.mean(vals)) if vals else 0.0

    # Turn into DataFrame (rows = groups, single column = metric)
    df = pd.DataFrame.from_dict(data, orient="index", columns=[metric])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(4, len(df)*0.8), 3))
    cax = ax.matshow(df.T, cmap="YlGnBu")
    fig.colorbar(cax, ax=ax)

    # Annotate cells
    for (i, j), v in np.ndenumerate(df.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black")

    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels([metric])
    plt.tight_layout()

    return jsonify({"image": fig_to_base64(fig)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
