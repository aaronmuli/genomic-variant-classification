import joblib
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from loader import Loader
from get_predictions import get_predictions

# ---- Same cleaning function ----
def clean_medical_text(text):
    text = str(text).lower()
    text = re.sub(r'\[[0-9,\-\s]+\]', '', text)
    text = re.sub(r'http\S+|www\S+|[^a-z\s]', '', text)
    text = " ".join(text.split())
    return text

loader = Loader("Loading model and vectorizer ")
loader.start()

# ---- Load model and vectorizer ----
pipeline = joblib.load("models/variant_pipeline.pkl")
model = pipeline["model"]
vectorizer = pipeline["vectorizer"]

loader.stop()


def get_top_medical_features(model, vectorizer, top_n=20):
    """
    Maps XGBoost importance scores back to medical keywords
    and returns the top N influencers as a dictionary.
    
    Returns:
        dict -> {keyword: gain_score}
    """
    # 1. Get importance scores using gain
    importance_scores = model.get_booster().get_score(importance_type='gain')
    
    # 2. Get feature names from TF-IDF
    feature_names = vectorizer.get_feature_names_out()
    
    # 3. Map 'f0', 'f1', ... to actual medical words
    mapped_importances = {}
    for f_id, score in importance_scores.items():
        index = int(f_id.replace('f', ''))
        if index < len(feature_names):  # safety check
            mapped_importances[feature_names[index]] = float(score)

    # 4. Sort by gain descending and keep top N
    sorted_items = sorted(
        mapped_importances.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    # 5. Convert to dictionary
    top_features_dict = dict(sorted_items)

    return top_features_dict


def predict_variant_class(clinical_text):
    # Clean text
    cleaned = clean_medical_text(clinical_text)

    # Convert to TF-IDF
    X_input = vectorizer.transform([cleaned])

    # Predict probabilities
    probs = model.predict_proba(X_input)[0]

    # Get predicted class (convert back to 1-9)
    predicted_class = probs.argmax() + 1

    confidence = probs[predicted_class - 1] * 100

    all_preds = get_predictions(probs=probs)

    medical_features = get_top_medical_features(model, vectorizer)

    return predicted_class, confidence, all_preds, medical_features


# ---- Example usage ----
if __name__ == "__main__":
    sample_text = """
    Patient displays a point mutation in TP53. Studies show this variant leads to a gain of function...
    """

    prediction, confidence, all_preds, medical_features = predict_variant_class(sample_text)

    print(f"\n🧬 Predicted Genetic Class: {prediction}")
    print(f"Model Top Confidence Score: {confidence:.2f}%")
    print("Class Probabilities:", all_preds)
    print("Top 20 medical features:", medical_features)
