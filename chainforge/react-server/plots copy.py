# plots.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

app = Flask(__name__)
CORS(app)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def compute_metric(y_true, y_pred, metric):
    y_t = [int(v) for v in y_true]
    y_p = [int(v) for v in y_pred]

    if metric == "f1":
        return f1_score(y_t, y_p, average="weighted")
    elif metric == "fpr":
        tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
        return fp / (fp + tn) if (fp + tn) else 0.0
    elif metric == "fnr":
        tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
        return fn / (fn + tp) if (fn + tp) else 0.0
    elif metric == "roc_auc":
        return roc_auc_score(y_t, y_p) if len(set(y_t)) > 1 else 0.0
    else:
        return 0.0


@app.route("/metrics_heatmap", methods=["POST"])
def metrics_heatmap():
    payload    = request.json or {}
    responses  = payload.get("responses", [])
    metric     = payload.get("plot_type", "f1")  # e.g. "f1", "fpr", etc.
    group_key  = payload.get("target_group_key", "target_group")
    llm_key    = payload.get("llm_key", "LLM")

    # Dictionary: (group, llm) → list of y_true/y_pred
    matrix = {}

    for r in responses:
        # Get group from metavars
        group = r.get("metavars", {}).get(group_key)
        llm   = r.get("llm", {}).get("name", r.get("llm", "unknown"))

        if group is None:
            continue

        y_true = r.get("metavars", {}).get("true_label")
        y_pred = None
        items = r.get("eval_res", {}).get("items") or []
        if items:
            y_pred = items[0]

        if y_true is None or y_pred is None:
            continue

        key = (group, llm)
        matrix.setdefault(key, {"y_true": [], "y_pred": []})
        matrix[key]["y_true"].append(int(y_true))
        matrix[key]["y_pred"].append(int(y_pred))

    # Pivot into DataFrame
    data = {}
    for (group, llm), values in matrix.items():
        score = compute_metric(values["y_true"], values["y_pred"], metric)
        data.setdefault(group, {})[llm] = score

    df = pd.DataFrame.from_dict(data, orient="index").fillna(0.0)

    # Plot
    fig, ax = plt.subplots(figsize=(1.5 + len(df.columns)*0.9, 1.5 + len(df.index)*0.6))
    cax = ax.matshow(df.values, cmap="viridis" if metric in ("f1", "roc_auc") else "OrRd", vmin=0, vmax=1)
    fig.colorbar(cax, ax=ax)

    # Annotate
    for (i, j), val in np.ndenumerate(df.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.5 else "black")

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel("LLM")
    ax.set_ylabel("Target Group")
    ax.set_title(f"{metric.upper()} by Target Group and LLM")
    plt.tight_layout()

    return jsonify({"image": fig_to_base64(fig)})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
