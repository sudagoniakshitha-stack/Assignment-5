
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "clustered_books.csv"
MATRIX_PATH = ROOT_DIR / "models" / "book_tfidf_matrix.npz"
RESULTS_PATH = ROOT_DIR / "reports" / "model_evaluation.csv"

df = pd.read_csv(DATA_PATH).reset_index(drop=True)
tfidf_matrix = load_npz(MATRIX_PATH)

if len(df) != tfidf_matrix.shape[0]:
    raise ValueError(
        "Book count and TF-IDF matrix size do not match."
    )

if "Rating" not in df.columns or "Cluster" not in df.columns:
    raise ValueError("Rating or Cluster column is missing.")

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

if len(df) < 2:
    raise ValueError("At least two books are required.")

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_recommendations(book_index, method, top_n=5):
    similarities = cosine_similarity(
        tfidf_matrix[book_index],
        tfidf_matrix
    ).flatten()

    similarities[book_index] = -1

    candidates = np.arange(len(df))

    if method == "clustering":
        candidates = candidates[
            df["Cluster"].to_numpy()
            == df.loc[book_index, "Cluster"]
        ]

    if method == "content":
        scores = similarities[candidates]

    elif method == "clustering":
        scores = similarities[candidates]

    elif method == "hybrid":
        same_cluster = (
            df["Cluster"].to_numpy()
            == df.loc[book_index, "Cluster"]
        ).astype(float)

        scores = (
            0.8 * similarities[candidates]
            + 0.2 * same_cluster[candidates]
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    valid = candidates != book_index
    candidates = candidates[valid]
    scores = scores[valid]

    order = np.argsort(scores)[::-1][:top_n]

    return candidates[order]


def evaluate_recommendations(top_k=5, sample_size=300):
    rng = np.random.default_rng(42)

    rated_indices = df.index[df["Rating"].notna()].to_numpy()

    if len(rated_indices) < 2:
        raise ValueError("Not enough rated books for evaluation.")

    sample_indices = rng.choice(
        rated_indices,
        size=min(sample_size, len(rated_indices)),
        replace=False
    )

    methods = ["content", "clustering", "hybrid"]
    results = []

    print("Evaluating recommendation models...")
    print(f"Books in dataset: {len(df)}")
    print(f"Sampled books: {len(sample_indices)}")
    print(f"Top K: {top_k}")

    for method in methods:
        precision_values = []
        recall_values = []

        for book_index in sample_indices:
            recommendations = get_recommendations(
                book_index,
                method,
                top_n=top_k
            )

            recommended_ratings = df.loc[
                recommendations, "Rating"
            ]

            # Proxy relevance rule:
            # A recommended book is relevant if its rating is >= 4.0.
            # This is a dataset-based rule, not actual user preference.
            relevant_recommendations = (
                recommended_ratings.dropna() >= 4.0
            ).sum()

            total_relevant = (
                df["Rating"].dropna() >= 4.0
            ).sum()

            precision = (
                relevant_recommendations / len(recommendations)
                if len(recommendations) > 0
                else 0.0
            )

            recall = (
                relevant_recommendations / total_relevant
                if total_relevant > 0
                else 0.0
            )

            precision_values.append(precision)
            recall_values.append(recall)

        results.append({
            "Model": method,
            "Precision@5": np.mean(precision_values),
            "Recall@5": np.mean(recall_values),
            "Evaluation Type": "Rating-threshold proxy"
        })

    results_df = pd.DataFrame(results)

    print("\nRecommendation Evaluation Results:")
    print(results_df.to_string(index=False))

    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to: {RESULTS_PATH}")

    return results_df


def evaluate_rating_baseline():
    # Baseline: predict the mean rating for every rated book.
    # This is a rating-prediction baseline, not a personalized
    # recommendation model or a rating prediction from the TF-IDF model.

    ratings = df["Rating"].dropna().to_numpy()

    if len(ratings) == 0:
        print("RMSE cannot be calculated: no ratings available.")
        return None

    actual = ratings
    predicted = np.full(len(actual), actual.mean())

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print("\nRating Prediction Baseline:")
    print(f"Mean rating: {actual.mean():.4f}")
    print(f"RMSE: {rmse:.4f}")

    return rmse


if __name__ == "__main__":
    print("Audible Insights Model Evaluation")

    evaluate_recommendations(top_k=5, sample_size=300)
    evaluate_rating_baseline()

