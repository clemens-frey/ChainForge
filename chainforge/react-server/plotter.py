

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


app = Flask(__name__)
CORS(app)

@app.route("/plot", methods=["POST"])
def plot():

    #herlperss
    def plot_metric_heatmap(metric_name):
        """Plot a heatmap of the metric for each LLM and group."""
        df = json_to_dataframe(responses)
        matrix = compute_metric_matrix(df, metric_name)
        fig = plot_heatmap(matrix, f"{metric_name.upper()} Heatmap")
        return fig

    def plotMatrix(ax):
        """Plot confusion matrix."""
        cm = calculate_metrics(y_true,y_pred)['confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative (0)", "Positive (1)"])
        disp.plot(cmap='spring', ax=ax) 
        ax.set_xlabel("predicted label")
        ax.set_ylabel("true_label")


    def json_to_dataframe(responses):
        records = []
        for entry in responses:
            try:
                records.append({
                    "group": entry["metavars"]["group"],
                    "llm": entry["llm"][ "name"],
                    "y_true": int(entry["metavars"]["true_label"]),
                    "y_pred": int(entry["eval_res"]["items"][0]),
                    "y_pred_values": float(entry["responses"][0]) 

                })
            except (KeyError, ValueError):
                continue
        return pd.DataFrame(records)

    def calculate_metrics(y_true, y_pred):
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
            # Return metrics as a dictionary
            return {"f1": f1, "fpr": fpr, "fnr": fnr, "auc": auc, "acc":accuracy, "prec":precision, "recall":recall, "confusion_matrix": cm}
        except Exception as e:
            print("Metric calc error:", e)
            return {"f1": 0, "fpr": 0, "fnr": 0, "auc": 0}

    def compute_metric_matrix(df, metric: str):
        result = {}

        grouped = df.groupby(["llm", "group"])
        for (llm, group), subdf in grouped:
            if metric == "auc123":
                # For AUC, we need to pin the subgroup examples with random samples from the full dataset
                df_sample_group = subdf.sample(frac = 0.5)
                try:
                    result[(llm, group)] = calculate_pinned_auc(df_sample_group, subdf.sample(df_sample_group.shape[0]), 'y_true', 'y_pred', 'y_pred_values')
                except ValueError as e:
                    result[(llm, group)] = 0.0
                    print(f"AUC calculation error for {llm}, {group}: {e}")
            else:
                m = calculate_metrics(subdf["y_true"], subdf["y_pred"])
                result[(llm, group)] = m[metric]

        # Convert to matrix DataFrame
        matrix = pd.Series(result).unstack(fill_value=0).sort_index()
        return matrix
    
        
    def calculate_pinned_auc(df_subgroup_examples, df_full_dataset_examples, y_true, y_pred, y_pred_values):
        # Combine examples from subgroup and random samples from full dataset
        pinned_dataset = pd.concat([df_subgroup_examples, df_full_dataset_examples]) 
        # remove null values
        pinned_dataset = pinned_dataset.dropna(subset=[y_pred, y_pred_values])
        # Calculate ROC-AUC for the pinned dataset
        pinned_auc = roc_auc_score(pinned_dataset[y_true].astype(int), pinned_dataset[y_pred_values].astype(float))

        return pinned_auc
    

    def plot_heatmap2(matrix, metric):
        fig, ax = plt.subplots(figsize=(10, 4))
        #YlOrBr
        im = ax.imshow(matrix.values, cmap=sns.color_palette("rocket_r", as_cmap=True), vmin=0, vmax=1)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(matrix.index)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iat[i, j]
                color = "white" if val > 0.4 else "black"
                #color = "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
                
        ax.set_title(f"{metric.upper()} Heatmap")
        plt.colorbar(im, ax=ax, fraction=0.03)
        plt.tight_layout()
        return fig
    
    def plot_heatmap(matrix: pd.DataFrame, title: str) -> plt.Figure:
        """Plot a heatmap using seaborn with annotations and fixed colormap scale."""
        fig, ax = plt.subplots(figsize=(10, 4))

        sns.heatmap(
            matrix,
            annot=True,            # Show the values in each cell
            fmt=".2f",             # Format of annotations
            cmap="rocket_r",       # You can use any Seaborn palette (e.g. "coolwarm", "YlGnBu", etc.)
            vmin=0, vmax=1,        # Keep color scale consistent across plots
            linewidths=0.5,        # Thin lines between cells
            linecolor="white",
            cbar=True,             # Show color bar
            ax=ax
        )

        ax.set_title(f"{title.upper()} Heatmap")
        ax.set_xlabel("Group")
        ax.set_ylabel("LLM")
        plt.tight_layout()

        return fig
    



    def plot_metric_table(df):
        """Create a table plot showing FNR, FPR, F1, AUC, PREC, RECALL, ACC per LLM with highlights."""
        
        # Group by LLM and aggregate metrics
        rows = []
        for llm, group_df in df.groupby("llm"):
            m = calculate_metrics(group_df["y_true"], group_df["y_pred"])
            rows.append({
                "LLM": llm,
                "ACC":  m["acc"],
                "PREC": m["prec"],
                "RECALL": m["recall"],
                "FNR": m["fnr"],
                "FPR": m["fpr"],
                "F1": m["f1"],
                "AUC": m["auc"],
            })

        # Create DataFrame for table
        table_df = pd.DataFrame(rows).set_index("LLM")

        # Prepare colors
        red = "#f08080"
        blue = "#add8e6"

        # Find best and worst for each column
        best_values = {}
        worst_values = {}
        for col in table_df.columns:
            ascending = col in ["FNR", "FPR"]  # Lower is better
            best_values[col] = table_df[col].min() if ascending else table_df[col].max()
            worst_values[col] = table_df[col].max() if ascending else table_df[col].min()

        # Plot as table
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

        # Highlight cells
        for i, llm in enumerate(table_df.index):
            for j, col in enumerate(table_df.columns):
                cell = table[i + 1, j]  # +1 for header row
                #cell.visible_edges = 'horizontal'
                #cell.set_edgecolor("white")
                value = table_df.loc[llm, col]
                if value == best_values[col]:
                    cell.set_facecolor(blue)
                elif value == worst_values[col]:
                    cell.set_facecolor(red)

        return fig


    def plot_metric_table4(df):
        """Create a table plot showing FNR, FPR, F1, AUC, PREC, RECALL, ACC per LLM with highlights and no visible vertical lines."""

        # Group by LLM and aggregate metrics
        rows = []
        for llm, group_df in df.groupby("llm"):
            m = calculate_metrics(group_df["y_true"], group_df["y_pred"])
            rows.append({
                "LLM": llm,
                "ACC":  round(m["acc"], 2),
                "PREC": round(m["prec"], 2),
                "RECALL": round(m["recall"], 2),
                "FNR": round(m["fnr"], 2),
                "FPR": round(m["fpr"], 2),
                "F1":  round(m["f1"], 2),
                "AUC": round(m["auc"], 2),
            })

        # Create DataFrame for table
        table_df = pd.DataFrame(rows).set_index("LLM")

        # Prepare colors
        red = "#f08080"
        blue = "#add8e6"

        # Identify best and worst values per column
        best_values = {}
        worst_values = {}
        for col in table_df.columns:
            ascending = col in ["FNR", "FPR"]  # Lower is better
            best_values[col] = table_df[col].min() if ascending else table_df[col].max()
            worst_values[col] = table_df[col].max() if ascending else table_df[col].min()

        # Plot as table
        fig, ax = plt.subplots(figsize=(8, 0.5 + 0.5 * len(table_df)))
        ax.axis("off")
        table = ax.table(cellText=table_df.values,
                        colLabels=table_df.columns,
                        rowLabels=table_df.index,
                        loc="center",
                        cellLoc="center")
        table.scale(1, 1.5)
        ax.set_title("Metrics by LLM", fontsize=14, pad=10)

        # Style cells
        for key, cell in table.get_celld().items():
            i, j = key
            cell.set_edgecolor("white")  # remove vertical lines visually
            cell.set_linewidth(2)      # optional: adjust line thickness
            if i == 0 and j > 0:  # header row, excluding index label
                #cell.set_edgecolor("white")  # first reset all
                cell.set_edgecolor("black")
                cell.set_linewidth(2)
                cell.visible_edges = "B"             # only bottom edge

            if i > 0 and j > 0:  # only data cells
                llm = table_df.index[i - 1]
                col = table_df.columns[j - 1]
                value = table_df.loc[llm, col]
                if value == best_values[col]:
                    cell.set_facecolor(blue)
                elif value == worst_values[col]:
                    cell.set_facecolor(red)

        return fig



    def plot_all_confusion_matrices(df):
        llms = df["llm"].unique()
        n = len(llms)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

        if n == 1:
            axes = [axes]

        for ax, llm in zip(axes, llms):
            subdf = df[df["llm"] == llm]
            cm = confusion_matrix(subdf["y_true"], subdf["y_pred"], labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative (0)", "Positive (1)"])
            disp.plot(cmap=sns.color_palette("rocket_r", as_cmap=True), ax=ax, colorbar=True)
            ax.set_title(llm)
            ax.set_xlabel("predicted label")
            ax.set_ylabel("true_label")

        plt.tight_layout()
        return fig
    


    def plot_all_confusion_matrices2(df):
        llms = df["llm"].unique()
        n = len(llms)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        
        if n == 1:
            axes = [axes]

        for ax, llm in zip(axes, llms):
            subdf = df[df["llm"] == llm]
            cm = confusion_matrix(subdf["y_true"], subdf["y_pred"], labels=[0, 1])
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='rocket_r',
                xticklabels=["Negative (0)", "Positive (1)"],
                yticklabels=["Negative (0)", "Positive (1)"],
                ax=ax,
                cbar=True
            )
            
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(llm)

        plt.tight_layout()
        return fig




    def fig_to_base64(fig: plt.Figure) -> str:
        """Serialize a Matplotlib figure to a base64‑encoded PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    
   
    #data = request.json

    payload      = request.json or {}
    llm_group    = payload.get("llm_group", "LLM") # e.g. "LLM" or "__meta_0"
    plot_type    = payload.get("plot_type", "line")
    responses    = payload.get("responses", [])


    y_true = [int(entry["metavars"]["true_label"]) for entry in payload["responses"]]
    y_pred = [entry["eval_res"]["items"][0] for entry in payload["responses"]]
    y_pred_values = [float(entry["responses"][0]) for entry in payload["responses"]]
      
      

    json_object = json.dumps(payload, indent=4)
        # Writing to sample.json
    with open("sample2.json", "w") as outfile:
        outfile.write(json_object)

    # create plot
    
    if plot_type == "matrix":
        fig, ax = plt.subplots()
        plotMatrix(ax)
    elif plot_type in ("f1", "fpr", "fnr", "auc"):
        df = json_to_dataframe(responses)
        matrix = compute_metric_matrix(df, plot_type)
        fig = plot_heatmap(matrix,plot_type)
    elif plot_type == "table":
        df = json_to_dataframe(responses)
        fig = plot_metric_table(df)
    elif plot_type == "confusion_per_llm":
        df = json_to_dataframe(responses)
        fig = plot_all_confusion_matrices(df)

    else:
        return jsonify({"error": f"Unsupported plot_type '{plot_type}'"}), 400
  
    # return json for React
    img_base64 = fig_to_base64(fig)
    return jsonify({"image": img_base64})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)


