# test_plot_server.py

import requests
import base64
from io import BytesIO
from PIL import Image

def test_plot():
    url = "http://localhost:5000/plot"
    payload = {
        # sample data: 5 points
        "x": ["A", "B", "A", "C", "B"],
        "y": [1.2, 2.5, 1.8, 3.3, 2.1],
        "plot_type": "bar"   # try "line", "scatter", "bar"
    }

    print(f"POSTing to {url} …")
    resp = requests.post(url, json=payload)
    resp.raise_for_status()

    data = resp.json()
    img_b64 = data.get("image")
    if not img_b64:
        raise RuntimeError("No ‘image’ field in JSON response")

    # Decode and display/save the image
    img_data = base64.b64decode(img_b64)
    buf = BytesIO(img_data)
    img = Image.open(buf)

    # Show it (requires a GUI-capable environment)
    img.show(title="Test Plot")

    #—or save to disk—
    out_path = "test_plot.png"
    img.save(out_path)
    print(f"Saved test plot to {out_path}")

if __name__ == "__main__":
    test_plot()
