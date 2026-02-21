import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loader import Loader
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import re
import joblib
import os

# --- PART 1: DATA LOADING AND MERGING FUNCTION ---
def load_and_merge_data(variants_file, text_file):
    loader = Loader("Loading dataset ")
    loader.start()

    # 1. Load the Genetic Variants
    # This contains the Gene name, the specific Mutation (Variation), and the Class (1-9)
    variants = pd.read_csv(variants_file)

    # 2. Load the Clinical Evidence (Text)
    # This is a large file containing medical papers related to the variants
    text = pd.read_csv(text_file, sep=r'\|\|', engine='python', 
                    header=None, skiprows=1, names=["ID", "Clinical_Evidence"])

    # 3. Merge them on the 'ID' column
    # This gives you a dataset where each mutation is linked to clinical papers
    df = pd.merge(variants, text, on='ID')

    loader.stop()
    return df

    # Visualize the distribution of the 9 Genetic Classes
    # plt.figure(figsize=(10,6))
    # sns.countplot(x='Class', data=df, palette='viridis')
    # plt.title('Distribution of Cancer Genetic Variants (Classes 1-9)')
    # plt.xlabel('Clinical Class')
    # plt.ylabel('Number of Variants')
    # plt.show()


# --- PART 2: THE MEDICAL SCRUB FUNCTION ---
def clean_medical_text(text):
    """Removes noise like citations [1] and URLs from medical papers."""
    text = str(text).lower()
    text = re.sub(r'\[[0-9,\-\s]+\]', '', text) # Remove citations
    text = re.sub(r'http\S+|www\S+|[^a-z\s]', '', text) # Remove URLs & symbols
    text = " ".join(text.split())

    return text

# --- PART 3: FEATURE EXTRACTION FUNCTION ---
def extract_tfidf_features(df):
    loader = Loader("Extracting TF-IDF features ")
    loader.start()

    """Turns cleaned text into numerical features for the AI."""
    df['Cleaned_Text'] = df['Clinical_Evidence'].apply(clean_medical_text)
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['Cleaned_Text'])

    loader.stop()
    return X, vectorizer

# --- PART 4: TRAINING FUNCTION ---
def train_variant_predictor(X, y):
    loader = Loader("Training the XGBoost model ")
    loader.start()
    # 1. Split data (80% for training, 20% for testing)
    # 'stratify=y' ensures both sets have a fair mix of all 9 cancer classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Initialize the XGBoost Super-Engine
    # objective='multi:softprob' tells the AI to give us probabilities for all 9 classes
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        objective='multi:softprob',
        eval_metric='mlogloss', # Medical Log-Loss is our scorecard
    )
    
    # In the MSK dataset, classes are 1-9. Python needs 0-8.
    model.fit(X_train, y_train - 1) 
    
    # 3. Evaluate the "Digital Pathologist"
    preds_proba = model.predict_proba(X_test)
    loss = log_loss(y_test - 1, preds_proba)
    
    loader.stop()
    print(f"Log Loss Score: {loss:.4f} (Lower is better)")
    return model


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # 1. Load Data
    data = load_and_merge_data('datasets/training_variants', 'datasets/training_text')
    
    # 2. Process Data
    X_features, medical_vectorizer = extract_tfidf_features(data)
    y_labels = data['Class']
    
    # 3. Train the model
    trained_model = train_variant_predictor(X_features, y_labels)

    # 4. Save model and vectorizer
    joblib.dump({
        "model": trained_model,
        "vectorizer": medical_vectorizer
    }, "models/variant_pipeline.pkl")


    print("✅ Model and vectorizer saved successfully in /models folder")
