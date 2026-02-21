import numpy as np

def get_predictions(probs):
    # Get indices of 9 probabilities (sorted descending)
    top3_indices = np.argsort(probs)[-9:][::-1]
    predictions = []

    for rank, idx in enumerate(top3_indices, start=1):
        probability = probs[idx]
        confidence = float(probs[idx]) * 100
        predictions.append({
            "rank": rank,
            "class": int(idx + 1),
            "confidence": confidence,
            "probability": float(probability)
        })

    return predictions
