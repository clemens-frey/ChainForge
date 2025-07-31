from flask import Flask, request, jsonify
from flask_cors import CORS 
import matplotlib.pyplot as plt
import io
import base64
import json
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score



app = Flask(__name__)
CORS(app)

@app.route("/plot", methods=["POST"])
def plot():
    #data = request.json

    payload      = request.json or {}
    selected_var = payload.get("selected_var")     # e.g. "accuracy" or "__meta_tag"
    llm_group    = payload.get("llm_group", "LLM") # e.g. "LLM" or "__meta_0"
    plot_type    = payload.get("plot_type", "line")
    responses    = payload.get("responses", [])    # list of LLMResponse dicts

    json_object = json.dumps(payload, indent=4)

    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)

    y_true = [int(entry["metavars"]["true_label"]) for entry in payload["responses"]]
    y_pred = [entry["eval_res"]["items"][0] for entry in payload["responses"]]

#    print("Paypload:" + str(payload))
#    print("Meta:" + str(payload.metavars))
#    print("text:" + str(payload.text))
#    print("prompt:" + str(payload.prompt))
#    print("llm:" + str(payload.llm))
#    print("var:" + str(payload.var))




    x_vals = []
    y_vals = []

    def get_llm_name(resp):
        # match your frontend logic: if group is "LLM", use resp["llm"]
        if llm_group == "LLM":
            llm = resp.get("llm")
            # could be a string or dict
            return llm if isinstance(llm, str) else llm.get("name", "unknown")
        # otherwise pull from resp["metavars"]
        return resp.get("metavars", {}).get(llm_group, "unknown")

    def get_y_value(resp):
        # first try resp["vars"][selected_var]
        vars_dict = resp.get("vars", {})
        raw       = vars_dict.get(selected_var)
        if raw is not None:
            try:
                return float(raw)
            except ValueError:
                pass
        # fallback: if eval_res exists and has numeric items, average them
        eval_res = resp.get("eval_res", {})
        items    = eval_res.get("items") or []
        nums     = [v for v in items if isinstance(v, (int, float))]
        if nums:
            return sum(nums) / len(nums)
        return 0.0

    for resp in responses:
        x_vals.append(get_llm_name(resp))
        y_vals.append(get_y_value(resp))

    # create plot
    fig, ax = plt.subplots()

    if plot_type == "matrix":
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative (0)", "Positive (1)"])
        disp.plot(cmap='Blues', ax=ax) 
    else:  # bar, default
        ax.bar(x_vals, y_vals)

    ax.set_xlabel("predicted_label")
    ax.set_ylabel("true_label")
    ax.set_title(f"{plot_type.title()} ") #of {selected_var} by {llm_group}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    #  to png and base64 
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # return json for React
    return jsonify({"image": img_base64})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

'''
    x = data.get("x", [])
    y = data.get("y", [])
    plot_type = data.get("plot_type", "line")

    fig, ax = plt.subplots()

    if plot_type == "bar":
        ax.bar(x, y)
    else:
        ax.plot(x, y)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return jsonify({"image": img_base64})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
'''
