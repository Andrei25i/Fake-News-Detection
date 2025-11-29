import os
import pickle
import torch
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification

from config import BERT_PATH, CLASSIC_PATH, DEVICE
from utils import clean_text_classic, clean_text_bert

artifacts = {
    "bert_model": None,
    "bert_tokenizer": None,
    "classic_model": None,
    "classic_vectorizer": None,
    "stop_words": None
}

def load_models():
    """Loads all AI models into memory/GPU on startup."""
    print("Services: Initializing AI models...")

    # Setup NLTK
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        artifacts["stop_words"] = set(stopwords.words('english'))
        print("NLTK Loaded.")
    except Exception as e:
        print(f"NLTK Error: {e}")

    # Load Classic
    try:
        model_file = os.path.join(CLASSIC_PATH, "model_lr.pkl")
        vect_file = os.path.join(CLASSIC_PATH, "tfidf_vectorizer.pkl")

        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                artifacts["classic_model"] = pickle.load(f)
            with open(vect_file, 'rb') as f:
                artifacts["classic_vectorizer"] = pickle.load(f)
            print("Classic Model loaded.")
        else:
            print("Classic model files missing.")
    except Exception as e:
        print(f"Failed to load Classic Model: {e}")

    # Load BERT
    try:
        if os.path.exists(BERT_PATH):
            artifacts["bert_tokenizer"] = BertTokenizer.from_pretrained(BERT_PATH)
            model = BertForSequenceClassification.from_pretrained(BERT_PATH)
            model.to(DEVICE)
            model.eval()
            artifacts["bert_model"] = model
            print(f"BERT loaded on {DEVICE}")
        else:
            print(f"BERT path missing: {BERT_PATH}")
    except Exception as e:
        print(f"Failed to load BERT: {e}")

async def predict(title: str, text: str):
    """Routing logic for prediction."""
    full_text = title + " " + text
    results = []

    # Predict with Classic model
    try:
        fake_prob, real_prob, model_name = _predict_classic(full_text)
        confidence = max(fake_prob, real_prob)
        prediction = "REAL" if real_prob > fake_prob else "FAKE"
        results.append({
            "model_used": model_name,
            "prediction": prediction,
            "confidence_score": round(confidence, 4),
            "confidence_percent": f"{confidence * 100:.2f}%",
            "probabilities": {
                "fake": round(fake_prob, 4),
                "real": round(real_prob, 4)
            }
        })
    except Exception as e:
        print(f"Classic model error: {e}")

    # Predict with BERT model
    try:
        fake_prob, real_prob, model_name = _predict_bert(full_text)
        confidence = max(fake_prob, real_prob)
        prediction = "REAL" if real_prob > fake_prob else "FAKE"
        results.append({
            "model_used": model_name,
            "prediction": prediction,
            "confidence_score": round(confidence, 4),
            "confidence_percent": f"{confidence * 100:.2f}%",
            "probabilities": {
                "fake": round(fake_prob, 4),
                "real": round(real_prob, 4)
            }
        })
    except Exception as e:
        print(f"BERT model error: {e}")

    return results

def _predict_classic(text: str):
    model = artifacts["classic_model"]
    vectorizer = artifacts["classic_vectorizer"]
    stop_words = artifacts["stop_words"]

    if not model: raise Exception("Classic model not active.")

    clean_content = clean_text_classic(text, stop_words)
    vectorized_text = vectorizer.transform([clean_content])

    probs = model.predict_proba(vectorized_text)[0]

    return probs[0], probs[1], "Logistic Regression Model"

def _predict_bert(text: str):
    model = artifacts["bert_model"]
    tokenizer = artifacts["bert_tokenizer"]

    if not model: raise Exception("BERT model not active.")

    clean_content = clean_text_bert(text)
    encoded = tokenizer(clean_content, padding=True, truncation=True, max_length=256, return_tensors='pt')

    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
    
    return probs[0][0].item(), probs[0][1].item(), "Bert Base Model"