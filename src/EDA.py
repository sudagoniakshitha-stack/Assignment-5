from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "cleaned_books.csv"
)

FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print("EDA: DATASET LOADED")
    print("Dataset shape:", df.shape)

    return df


def analyze_data(df):
    print("\nDATASET OVERVIEW")
    print(df.info())

    print("\nNUMERICAL SUMMARY")
    print(
        df[
            [
                "Rating",
                "Number of Reviews",
                "Price",
                "Listening Time"
            ]
        ].describe()
    )

    print("\nMISSING VALUES")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTOP 10 HIGHEST-RATED BOOKS")

    top_rated = (
        df.dropna(subset=["Rating"])
        .sort_values(
            ["Rating", "Number of Reviews"],
            ascending=[False, False]
        )
        [["Book Name", "Author", "Rating", "Number of Reviews"]]
        .head(10)
    )

    print(top_rated.to_string(index=False))

    print("\nTOP 10 MOST-REVIEWED BOOKS")

    most_reviewed = (
        df.dropna(subset=["Number of Reviews"])
        .sort_values(
            "Number of Reviews",
            ascending=False
        )
        [["Book Name", "Author", "Number of Reviews", "Rating"]]
        .head(10)
    )

    print(most_reviewed.to_string(index=False))

    print("\nTOP 10 MOST EXPENSIVE BOOKS")

    expensive = (
        df.dropna(subset=["Price"])
        .sort_values(
            "Price",
            ascending=False
        )
        [["Book Name", "Author", "Price", "Rating"]]
        .head(10)
    )

    print(expensive.to_string(index=False))


def plot_rating_distribution(df):
    ratings = df["Rating"].dropna()

    plt.figure(figsize=(8, 5))

    sns.histplot(
        ratings,
        bins=20,
        kde=True
    )

    plt.title("Distribution of Book Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Number of Books")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "rating_distribution.png",
        dpi=300
    )

    plt.close()


def plot_price_distribution(df):
    prices = df["Price"].dropna()

    plt.figure(figsize=(8, 5))

    sns.histplot(
        prices,
        bins=30,
        kde=True
    )

    plt.title("Distribution of Book Prices")
    plt.xlabel("Price")
    plt.ylabel("Number of Books")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "price_distribution.png",
        dpi=300
    )

    plt.close()


def plot_reviews_distribution(df):
    reviews = df["Number of Reviews"].dropna()

    plt.figure(figsize=(8, 5))

    sns.histplot(
        reviews,
        bins=30
    )

    plt.title("Distribution of Number of Reviews")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Number of Books")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "reviews_distribution.png",
        dpi=300
    )

    plt.close()


def plot_missing_values(df):
    missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values to plot.")
        return

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=missing.values,
        y=missing.index
    )

    plt.title("Missing Values by Column")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Column")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "missing_values.png",
        dpi=300
    )

    plt.close()


def plot_listening_time_distribution(df):
    listening_time = df["Listening Time"].dropna()

    if listening_time.empty:
        print(
            "No listening-time values available for plotting."
        )
        return

    plt.figure(figsize=(8, 5))

    sns.histplot(
        listening_time,
        bins=30,
        kde=True
    )

    plt.title("Distribution of Audiobook Listening Time")
    plt.xlabel("Listening Time (minutes)")
    plt.ylabel("Number of Books")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "listening_time_distribution.png",
        dpi=300
    )

    plt.close()
def extract_genres(value):
    if pd.isna(value):
        return []

    pattern = r"#\s*[\d,]+\s+in\s+(.*?)(?=,\s*#\s*[\d,]+\s+in\s+|$)"

    matches = re.findall(pattern, str(value))

    genres = []

    for genre in matches:
        genre = genre.strip().strip(",")

        if not genre:
            continue

        if genre.startswith("Audible Audiobooks & Originals"):
            continue

        genres.append(genre)

    return genres


def analyze_genres(df):
    print("\nTOP 15 MOST POPULAR GENRES")

    genre_df = df[["Book Name", "Ranks and Genre"]].copy()

    genre_df["Genre"] = genre_df["Ranks and Genre"].apply(
        extract_genres
    )

    genre_df = genre_df.explode("Genre")

    genre_df = genre_df.dropna(subset=["Genre"])

    genre_counts = genre_df["Genre"].value_counts()

    print(genre_counts.head(15).to_string())

    return genre_df


def plot_popular_genres(genre_df):

    top_genres = (
        genre_df["Genre"]
        .value_counts()
        .head(15)
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        x=top_genres.values,
        y=top_genres.index
    )

    plt.title("Top 15 Most Popular Genres")
    plt.xlabel("Number of Books")
    plt.ylabel("Genre")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "popular_genres.png",
        dpi=300
    )

    plt.close()


def analyze_top_authors(df):

    print("\nTOP 10 AUTHORS BY AVERAGE RATING")

    author_stats = (
        df.dropna(subset=["Rating"])
        .groupby("Author")
        .agg(
            Average_Rating=("Rating", "mean"),
            Total_Books=("Book Name", "count"),
            Total_Reviews=("Number of Reviews", "sum")
        )
        .reset_index()
    )

    # Consider authors with at least 3 rated books
    author_stats = author_stats[
        author_stats["Total_Books"] >= 3
    ]

    top_authors = author_stats.sort_values(
        ["Average_Rating", "Total_Books"],
        ascending=[False, False]
    ).head(10)

    print(top_authors.to_string(index=False))

    return author_stats


def plot_top_authors(author_stats):

    top_authors = author_stats.sort_values(
        "Average_Rating",
        ascending=False
    ).head(10)

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=top_authors,
        x="Average_Rating",
        y="Author"
    )

    plt.title("Top 10 Authors by Average Rating")
    plt.xlabel("Average Rating")
    plt.ylabel("Author")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "top_authors_by_rating.png",
        dpi=300
    )

    plt.close()


def plot_rating_vs_reviews(df):

    plot_data = df.dropna(
        subset=["Rating", "Number of Reviews"]
    ).copy()

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=plot_data,
        x="Number of Reviews",
        y="Rating",
        alpha=0.5
    )

    plt.title("Book Ratings vs Number of Reviews")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Rating")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "rating_vs_reviews.png",
        dpi=300
    )

    plt.close()


def plot_author_popularity(author_stats):

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=author_stats,
        x="Total_Reviews",
        y="Average_Rating",
        size="Total_Books",
        sizes=(30, 200),
        alpha=0.6
    )

    plt.title("Author Popularity vs Average Rating")
    plt.xlabel("Total Number of Reviews")
    plt.ylabel("Average Rating")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "author_popularity_vs_rating.png",
        dpi=300
    )

    plt.close()


def plot_rating_by_genre(df, genre_df):

    rating_genres = genre_df.merge(
        df[["Book Name", "Rating"]],
        on="Book Name",
        how="left"
    )

    top_genres = (
        rating_genres["Genre"]
        .value_counts()
        .head(10)
        .index
    )

    rating_genres = rating_genres[
        rating_genres["Genre"].isin(top_genres)
    ].dropna(subset=["Rating"])

    plt.figure(figsize=(13, 7))

    sns.boxplot(
        data=rating_genres,
        x="Rating",
        y="Genre"
    )

    plt.title("Rating Distribution Across Top 10 Genres")
    plt.xlabel("Book Rating")
    plt.ylabel("Genre")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "rating_by_genre.png",
        dpi=300
    )

    plt.close()
def plot_correlation_heatmap(df):
    numerical_data = df[
        ["Rating", "Number of Reviews", "Price", "Listening Time"]
    ]
    correlation = numerical_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()
def analyze_clusters():
    cluster_df = pd.read_csv(
        PROJECT_DIR / "data" / "processed" / "clustered_books.csv"
    )

    counts = cluster_df["Cluster"].value_counts().sort_index()

    print("\nBOOKS IN EACH CLUSTER")
    print(counts)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Books in Each Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Books")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "cluster_distribution.png")
    plt.close()
def analyze_hidden_gems(df):
    data = df.dropna(subset=["Rating", "Number of Reviews"]).copy()

    review_limit = data["Number of Reviews"].median()

    hidden_gems = data[
        (data["Rating"] >= 4.5) &
        (data["Number of Reviews"] <= review_limit)
    ]

    hidden_gems = hidden_gems.sort_values(
        ["Rating", "Number of Reviews"],
        ascending=[False, True]
    ).head(20)

    print("\nTOP 20 HIDDEN GEMS")
    print(hidden_gems[
        ["Book Name", "Author", "Rating", "Number of Reviews"]
    ].to_string(index=False))

    hidden_gems.to_csv(
        PROJECT_DIR / "reports" / "hidden_gems.csv",
        index=False
    )
def recommend_by_genre(genre_name):
    file_path = PROJECT_DIR / "data" / "processed" / "clustered_books.csv"
    books = pd.read_csv(file_path)

    books["Genre_List"] = books["Ranks and Genre"].apply(extract_genres)

    recommendations = books[
        books["Genre_List"].apply(
            lambda genres: any(genre_name.lower() in g.lower() for g in genres)
        )
    ]

    recommendations = recommendations.dropna(subset=["Rating"])

    recommendations = recommendations.sort_values(
        "Rating", ascending=False
    ).head(5)

    print(f"\nTOP 5 BOOKS FOR {genre_name.upper()}")

    print(recommendations[
        ["Book Name", "Author", "Rating"]
    ].to_string(index=False))

def main():
    df = load_data()

    analyze_data(df)

    plot_rating_distribution(df)
    plot_price_distribution(df)
    plot_reviews_distribution(df)
    plot_missing_values(df)
    plot_listening_time_distribution(df)

    genre_df = analyze_genres(df)

    plot_popular_genres(genre_df)

    author_stats = analyze_top_authors(df)

    plot_top_authors(author_stats)

    plot_rating_vs_reviews(df)

    plot_author_popularity(author_stats)

    plot_rating_by_genre(df, genre_df)
    plot_correlation_heatmap(df)
    analyze_clusters()
    analyze_hidden_gems(df)
    recommend_by_genre("Science Fiction")
    recommend_by_genre("Thriller")
    print("\nEDA COMPLETED")
    print("Charts saved in:", FIGURES_DIR)


if __name__ == "__main__":
    main()