
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "clustered_books.csv"
VECTOR_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.joblib"
MATRIX_PATH = ROOT_DIR / "models" / "book_tfidf_matrix.npz"

df = pd.read_csv(DATA_PATH)
tfidf_matrix = load_npz(MATRIX_PATH)
vectorizer = joblib.load(VECTOR_PATH)

df = df.reset_index(drop=True)

if len(df) != tfidf_matrix.shape[0]:
    raise ValueError(
        "The number of books does not match the TF-IDF matrix. "
        "Check that both files were generated from the same dataset."
    )

required_columns = ["Book Name", "Author"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column not found: {column}")


def find_book(title):
    matches = df[
        df["Book Name"].astype(str).str.contains(
            title, case=False, na=False, regex=False
        )
    ]

    if matches.empty:
        print(f"No book found matching '{title}'.")
        return None

    return matches.index[0]


def get_recommendations(book_index, method="content", top_n=5):
    if book_index < 0 or book_index >= len(df):
        raise IndexError("Invalid book index.")

    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    similarity_scores = cosine_similarity(
        tfidf_matrix[book_index],
        tfidf_matrix
    ).flatten()

    similarity_scores[book_index] = -1

    result = df.copy()
    result["Text Similarity"] = similarity_scores

    if method == "content":
        result["Recommendation Score"] = result["Text Similarity"]

    elif method == "clustering":
        if "Cluster" not in result.columns:
            raise ValueError(
                "Cluster column is missing from clustered_books.csv."
            )

        target_cluster = result.loc[book_index, "Cluster"]

        result = result[result["Cluster"] == target_cluster].copy()

        result["Recommendation Score"] = result["Text Similarity"]

    elif method == "hybrid":
        if "Cluster" not in result.columns:
            raise ValueError(
                "Cluster column is missing from clustered_books.csv."
            )

        target_cluster = result.loc[book_index, "Cluster"]

        result["Same Cluster"] = (
            result["Cluster"] == target_cluster
        ).astype(int)

        result["Recommendation Score"] = (
            0.8 * result["Text Similarity"]
            + 0.2 * result["Same Cluster"]
        )

    else:
        raise ValueError(
            "Invalid method. Choose 'content', 'clustering', or 'hybrid'."
        )

    result = result[result.index != book_index]

    result = result.sort_values(
        by="Recommendation Score",
        ascending=False
    )

    display_columns = [
        "Book Name",
        "Author",
        "Rating",
        "Recommendation Score"
    ]

    if "Cluster" in result.columns:
        display_columns.append("Cluster")

    return result[display_columns].head(top_n).reset_index(drop=True)


def recommend_by_title(title, method="content", top_n=5):
    book_index = find_book(title)

    if book_index is None:
        return None

    selected_book = df.loc[book_index]

    print("\nSelected book:")
    print(f"Title: {selected_book['Book Name']}")
    print(f"Author: {selected_book['Author']}")

    recommendations = get_recommendations(
        book_index=book_index,
        method=method,
        top_n=top_n
    )

    print(f"\n{method.title()}-Based Recommendations:")
    print(recommendations.to_string(index=False))

    return recommendations


if __name__ == "__main__":
    print("Audible Insights Recommendation System")
    print(f"Total books available: {len(df)}")

    example_book = df["Book Name"].dropna().iloc[0]

    recommend_by_title(
        title=example_book,
        method="content",
        top_n=5
    )

    recommend_by_title(
        title=example_book,
        method="clustering",
        top_n=5
    )

    recommend_by_title(
        title=example_book,
        method="hybrid",
        top_n=5
    )

