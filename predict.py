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


def plot_top_medical_features(model, vectorizer, top_n=20):
    """
    Maps XGBoost importance scores back to medical keywords
    and plots the top N influencers.
    """
    # 1. Get the importance scores (type='gain')
    # We use 'gain' because it reflects the importance for prediction accuracy
    importance_scores = model.get_booster().get_score(importance_type='gain')
    
    # 2. Get the actual medical words from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # 3. Map scores to words
    # XGBoost uses 'f0', 'f1' etc. so we map 'f0' -> feature_names[0]
    mapped_importances = {}
    for f_id, score in importance_scores.items():
        index = int(f_id.replace('f', ''))
        mapped_importances[feature_names[index]] = score

    # 4. Create a DataFrame for easy plotting
    df_imp = pd.DataFrame(
        list(mapped_importances.items()), 
        columns=['Medical_Keyword', 'Gain_Score']
    ).sort_values(by='Gain_Score', ascending=False)


    # 5. Plot the results
    plt.figure(figsize=(12, 8))
    plt.barh(df_imp['Medical_Keyword'].head(top_n), df_imp['Gain_Score'].head(top_n), color='skyblue')
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Medical Keywords Driving Cancer Classification", fontsize=15)
    plt.xlabel("Importance (Gain Score)", fontsize=12)
    plt.ylabel("Clinical Term / Gene", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


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

    plot_top_medical_features(model, vectorizer)

    return predicted_class, confidence, all_preds


# ---- Example usage ----
if __name__ == "__main__":
    sample_text = """
    Patient displays a point mutation in TP53. Studies show this variant leads to a gain of function...
    """

    prediction, confidence, all_preds = predict_variant_class(sample_text)

    print(f"\n🧬 Predicted Genetic Class: {prediction}")
    print(f"Model Top Confidence Score: {confidence:.2f}%")
    print("Class Probabilities:", all_preds)
