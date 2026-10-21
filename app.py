"""Web app and REST API for checking whether an SMS is spam."""
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

MAX_LENGTH = 1000


def create_app(model_path="model.joblib"):
    if not Path(model_path).exists():
        raise SystemExit("Model file not found. Run this first: python train.py")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    app = Flask(__name__)

    def classify(text):
        is_spam = bool(model.predict([text])[0])
        return {"text": text, "is_spam": is_spam, "label": "spam" if is_spam else "ham"}

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        if request.method == "POST":
            text = request.form.get("message", "").strip()[:MAX_LENGTH]
            if text:
                result = classify(text)
        return render_template("index.html", result=result, model_name=bundle["name"])

    @app.post("/api/predict")
    def predict():
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify(error="The 'text' field is required"), 400
        if len(text) > MAX_LENGTH:
            return jsonify(error=f"Text can be at most {MAX_LENGTH} characters long"), 400
        return jsonify(classify(text))

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
