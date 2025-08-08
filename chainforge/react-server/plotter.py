from flask import Flask, request, jsonify
from flask_cors import CORS 
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64
import json
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import seaborn as sns

import hashlib
import json

chached_plots: dict[str, str] = {}


app = Flask(__name__)
CORS(app)

@app.route("/plot", methods=["POST"])
def plot():
    #helpersss
    def floatable(val):
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False
        

    def json_to_dataframe(responses):
        records = []
        for entry in responses:
            try:
                meta = entry.get('metavars') or {}
                group = meta.get('group', 'ALL')
                true_label = int(meta.get('true_label', 0))
                pred_label = int(entry['eval_res']['items'][0])
                pred_values = float(entry['responses'][0]) if floatable(entry['responses'][0]) else 0.0
                llm = entry.get('llm')["name"]
                records.append({
                    "group": group,
                    "llm": llm,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pred_values": pred_values,

                })
            except (KeyError, ValueError):
                continue
        return pd.DataFrame(records)
    

    def calculate_pinned_auc(df_subgroup_examples, df_full_dataset_examples, true_label, pred_label, pred_values):
        # Combine examples from subgroup and random samples from full dataset
        pinned_dataset = pd.concat([df_subgroup_examples, df_full_dataset_examples]) 
        # remove null values
        pinned_dataset = pinned_dataset.dropna(subset=[pred_label, pred_values])
        # Calculate ROC-AUC for the pinned dataset
        pinned_auc = roc_auc_score(pinned_dataset[true_label].astype(int), pinned_dataset[pred_values].astype(float))

        return pinned_auc


    def calculate_metrics(df_group, df):
        try:
            true_label=df_group["true_label"]
            pred_label=df_group["pred_label"]
            pred_values=df_group["pred_values"]

            matrix = confusion_matrix(true_label, pred_label, labels=[0, 1])
            tn, fp, fn, tp = matrix.ravel()
            f1 = f1_score(true_label, pred_label, average="binary", zero_division=0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            roc_auc = roc_auc_score(true_label, pred_values) if len(set(true_label)) > 1 else 0.0
            pinned_auc = calculate_pinned_auc(df_group, df.sample(df_group.shape[0]), 'true_label', 'pred_label', 'pred_values')

            return {"f1": f1, "fpr": fpr, "fnr": fnr, "pinned auc": pinned_auc,"roc auc":roc_auc, "acc":accuracy, "prec":precision, "recall":recall, "confusion_matrix": matrix}
        except Exception as e:
            print("calulate_metrics error:", e)
            return {"f1": 0, "fpr": 0, "fnr": 0, "pinned auc": 0}



    def grouped_metrics(df, metric: str):
        result = {}
        grouped = df.groupby(["llm", "group"])
        for (llm, group), subdf in grouped:
            m = calculate_metrics(subdf,df)
            result[(llm, group)] = m[metric]

        # Convert to matrix DataFrameS
        matrix = pd.Series(result).unstack(fill_value=0).sort_index()
        return matrix
    
    def get_hash(data):
        data_string = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_string.encode()).hexdigest()
    
        #matplotlib fig to base64 PNG
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    
    # check if llm responded with confidence scores
    # if not, we cant calculate ROC AUC
    def includesScore(df: pd.DataFrame) -> bool:
        if "pred_values" not in df.columns:
            return False
        vals = df["pred_values"].dropna()
        return (vals != 0).any()


    #plottersss

    def plot_heatmap(df, plot_type):
        grouped = grouped_metrics(df, plot_type)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            grouped,
            annot=True,        
            fmt=".2f",            
            cmap="rocket_r",       
            vmin=0, vmax=1,        
            linewidths=0.5,        
            linecolor="white",
            cbar=True,             
            ax=ax                   
        )

        ax.set_title(f"{plot_type.upper()} Heatmap")
        ax.set_xlabel("Group")
        ax.set_ylabel("LLM")
        plt.tight_layout()
        return fig
    


    def plot_metric_table(df):       
        # metrics by LLM
        rows = []
        for llm, llm_df in df.groupby("llm"):
            m = calculate_metrics(llm_df, df)
            rows.append({
                "LLM": llm,
                "ACC":  m["acc"],
                "PREC": m["prec"],
                "RECALL": m["recall"],
                "FNR": m["fnr"],
                "FPR": m["fpr"],
                "F1": m["f1"],
                #"ROC AUC": m["roc auc"],
                **({"ROC AUC": m["roc auc"]} if includesScore(df) else {}),
            })
        table_df = pd.DataFrame(rows).set_index("LLM")
        best = {}
        worst = {}
        for col in table_df.columns:
            flipped = col in ["FNR", "FPR"]  # lower better
            best[col] = table_df[col].min() if flipped else table_df[col].max()
            worst[col] = table_df[col].max() if flipped else table_df[col].min()

        fig, ax = plt.subplots(figsize=(8, 0.5 + 0.5 * len(table_df)))
        ax.axis("off")
        percent_df = table_df.map(lambda x: f"{x * 100:.1f}%")

        table = ax.table(cellText=percent_df.values,
                        colLabels=table_df.columns,
                        rowLabels=table_df.index,
                        loc="center",
                        cellLoc="center")
        table.scale(1, 1.5)
        #table.set_edgecolor("white")
        ax.set_title("Metrics by LLM", fontsize=14, pad=10)
        red = "#f08080"
        blue = "#add8e6"
        # highlighttt
        for i, llm in enumerate(table_df.index):
            for j, col in enumerate(table_df.columns):
                cell = table[i + 1, j]  #skip header
                #cell.visible_edges = 'horizontal'
                value = table_df.loc[llm, col]
                if value == best[col]:
                    cell.set_facecolor(blue)
                elif value == worst[col]:
                    cell.set_facecolor(red)

        return fig



    def plot_all_matrix(df):
        llms = df["llm"].unique()
        n = len(llms)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, llm in zip(axes, llms):
            subdf = df[df["llm"] == llm]
            cm = confusion_matrix(subdf["true_label"], subdf["pred_label"], labels=[0, 1])
            #cm = calculate_metrics(subdf["true_label"],subdf["pred_label"],subdf)['confusion_matrix']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative (0)", "Positive (1)"])
            disp.plot(cmap=sns.color_palette("rocket_r", as_cmap=True), ax=ax, colorbar=True)
            ax.set_title(llm)
            ax.set_xlabel("predicted label")
            ax.set_ylabel("true_label")

        plt.tight_layout()
        return fig
    
    
   

    payload = request.json or {}
    plot_type = payload.get("plot_type", "line")
    responses = payload.get("responses", [])


    #chache to fix repeated plotting
    global chached_plots
    hash = get_hash(payload)
    if hash in chached_plots:
        return jsonify({"image": chached_plots[hash]})

    # create plot
    df = json_to_dataframe(responses)
    if plot_type in ("f1", "fpr", "fnr", "pinned auc","roc auc"):
        fig = plot_heatmap(df,plot_type)
    elif plot_type == "table":
        fig = plot_metric_table(df)
    elif plot_type == "confusion_per_llm":
        fig = plot_all_matrix(df)

    else:
        return jsonify({"error": f"no legit plot_type '{plot_type}'"}), 400
  
   

    img_base64 = fig_to_base64(fig)
    chached_plots[hash] = img_base64
    return jsonify({"image": img_base64})



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)


