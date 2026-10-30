from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "cleaned_books.csv"
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df


def prepare_text(df):
    df = df.copy()

    text_columns = [
        "Book Name",
        "Author",
        "Description",
        "Ranks and Genre"
    ]

    for column in text_columns:
        if column not in df.columns:
            df[column] = ""

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df["combined_text"] = (
        df["Book Name"] + " "
        + df["Author"] + " "
        + df["Description"] + " "
        + df["Ranks and Genre"]
    ).str.strip()

    empty_text = df["combined_text"].eq("")

    print("\nText preparation")
    print("Total books:", len(df))
    print("Books with empty text:", int(empty_text.sum()))

    return df


def create_tfidf_features(df):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["combined_text"]
    )

    print("\nTF-IDF features created.")
    print("Matrix shape:", tfidf_matrix.shape)

    return vectorizer, tfidf_matrix


def choose_cluster_count(tfidf_matrix):
    print("\nEvaluating candidate cluster counts.")

    results = []

    # Test several candidate values of K.
    for k in [3, 4, 5, 6, 7, 8]:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(tfidf_matrix)

        # Silhouette score can be expensive on large datasets.
        # Use a reproducible sample to keep evaluation practical.
        score = silhouette_score(
            tfidf_matrix,
            labels,
            sample_size=min(1000, tfidf_matrix.shape[0]),
            random_state=42
        )

        results.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette Score": score
        })

        print(
            f"K={k}: "
            f"Inertia={model.inertia_:.2f}, "
            f"Silhouette={score:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        PROCESSED_DIR / "clustering_scores.csv",
        index=False
    )

    # Save an elbow plot.
    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["K"],
        results_df["Inertia"],
        marker="o"
    )

    plt.title("Elbow Method for K-Means Clustering")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "kmeans_elbow.png",
        dpi=300
    )

    plt.close()

    # Select the candidate with the best silhouette score.
    best_k = int(
        results_df.loc[
            results_df["Silhouette Score"].idxmax(),
            "K"
        ]
    )

    print("\nSelected K based on silhouette score:", best_k)

    return best_k


def perform_clustering(df, tfidf_matrix, k):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df = df.copy()

    df["Cluster"] = model.fit_predict(tfidf_matrix)

    cluster_counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Number of Books")
    )

    print("\nCluster sizes:")
    print(cluster_counts.to_string(index=False))

    cluster_counts.to_csv(
        PROCESSED_DIR / "cluster_sizes.csv",
        index=False
    )

    # Save book metadata and cluster assignments.
    output_columns = [
        "Book Name",
        "Author",
        "Rating",
        "Number of Reviews",
        "Price",
        "Listening Time",
        "Ranks and Genre",
        "Cluster"
    ]

    if "Description" in df.columns:
        output_columns.insert(7, "Description")

    df[output_columns].to_csv(
        PROCESSED_DIR / "clustered_books.csv",
        index=False
    )

    return df, model


def save_tfidf_features(vectorizer, tfidf_matrix):
    import joblib
    from scipy.sparse import save_npz

    joblib.dump(
        vectorizer,
        MODELS_DIR / "tfidf_vectorizer.joblib"
    )

    save_npz(
        MODELS_DIR / "book_tfidf_matrix.npz",
        tfidf_matrix
    )

    print("\nSaved TF-IDF vectorizer and feature matrix.")


def main():
    df = load_data()

    df = prepare_text(df)

    vectorizer, tfidf_matrix = create_tfidf_features(df)

    best_k = choose_cluster_count(tfidf_matrix)

    clustered_df, kmeans_model = perform_clustering(
        df,
        tfidf_matrix,
        best_k
    )

    import joblib

    joblib.dump(
        kmeans_model,
        MODELS_DIR / "kmeans_model.joblib"
    )

    save_tfidf_features(
        vectorizer,
        tfidf_matrix
    )

    print("\nNLP AND CLUSTERING COMPLETED.")
    print("Clustered dataset saved to:")
    print(PROCESSED_DIR / "clustered_books.csv")
    print("Model files saved to:", MODELS_DIR)
    print("Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()