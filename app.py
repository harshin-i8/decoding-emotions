from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = BertConfig.from_pretrained("bert-base-uncased")
config.num_labels = len(emotion_labels)
config.problem_type = "multi_label_classification"
model = BertForSequenceClassification(config)

quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
quantized_model.load_state_dict(torch.load("quantized_model.pth", map_location=torch.device("cpu")))
quantized_model.eval()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = quantized_model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().tolist()

    emotion_scores = [
        {"label": label, "score": round(prob * 100, 2)}
        for label, prob in zip(emotion_labels, probs)
    ]
    emotion_scores.sort(key=lambda x: x["score"], reverse=True)
    top_emotion = emotion_scores[0] if emotion_scores else None

    return jsonify({
        "input_text": text,
        "top_emotion": top_emotion,
        "emotions": emotion_scores
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # default to 10000 as Render suggests
    app.run(host="0.0.0.0", port=port)
