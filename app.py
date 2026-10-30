from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "clustered_books.csv"
MATRIX_PATH = ROOT_DIR / "models" / "book_tfidf_matrix.npz"
VECTOR_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.joblib"

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
HIDDEN_GEMS_PATH = ROOT_DIR / "reports" / "hidden_gems.csv"

st.set_page_config(
    page_title="Audible Insights",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def load_resources():
    books = pd.read_csv(DATA_PATH).reset_index(drop=True)
    matrix = load_npz(MATRIX_PATH)
    vectorizer = joblib.load(VECTOR_PATH)

    if len(books) != matrix.shape[0]:
        raise ValueError(
            "Book dataset and TF-IDF matrix have different row counts."
        )

    return books, matrix, vectorizer


try:
    df, tfidf_matrix, vectorizer = load_resources()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


def extract_genres(value):
    if pd.isna(value):
        return []

    pattern = r"#\s*[\d,]+\s+in\s+(.*?)(?=,\s*#\s*[\d,]+\s+in\s+|$)"

    genres = re.findall(pattern, str(value))

    return [
        genre.strip()
        for genre in genres
        if genre.strip()
        and not genre.startswith("Audible Audiobooks & Originals")
    ]


def get_recommendations(book_index, method, top_n):

    similarities = cosine_similarity(
        tfidf_matrix[book_index],
        tfidf_matrix
    ).flatten()

    similarities[book_index] = -1

    candidates = np.arange(len(df))

    if method == "Content-Based":

        scores = similarities[candidates]

    elif method == "Clustering-Based":

        target_cluster = df.loc[book_index, "Cluster"]

        candidates = candidates[
            df["Cluster"].to_numpy() == target_cluster
        ]

        scores = similarities[candidates]

    elif method == "Hybrid":

        target_cluster = df.loc[book_index, "Cluster"]

        same_cluster = (
            df["Cluster"].to_numpy() == target_cluster
        ).astype(float)

        scores = (
            0.8 * similarities[candidates]
            + 0.2 * same_cluster[candidates]
        )

    else:
        raise ValueError("Invalid recommendation method.")

    valid = candidates != book_index

    candidates = candidates[valid]
    scores = scores[valid]

    order = np.argsort(scores)[::-1][:top_n]

    selected_indices = candidates[order]
    selected_scores = scores[order]

    results = df.iloc[selected_indices].copy()

    results["Recommendation Score"] = selected_scores

    columns = [
        "Book Name",
        "Author",
        "Rating",
        "Recommendation Score"
    ]

    if "Cluster" in results.columns:
        columns.append("Cluster")

    return results[columns].reset_index(drop=True)


def get_genre_recommendations(genre_name, top_n=5):

    books = df.copy()

    books["Genre_List"] = books["Ranks and Genre"].apply(
        extract_genres
    )

    results = books[
        books["Genre_List"].apply(
            lambda genres: any(
                genre_name.lower() in genre.lower()
                for genre in genres
            )
        )
    ]

    results = results.dropna(subset=["Rating"])

    results = results.sort_values(
        ["Rating", "Number of Reviews"],
        ascending=[False, False]
    )

    return results.head(top_n)[
        ["Book Name", "Author", "Rating", "Number of Reviews"]
    ].reset_index(drop=True)


def introduction_page():

    st.title("Audible Insights")
    st.subheader("Intelligent Book Recommendation System")

    st.write(
        "Discover books that match your interests using "
        "Natural Language Processing and Machine Learning."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Books", f"{len(df):,}")

    col2.metric(
        "Recommendation Models",
        "3"
    )

    col3.metric(
        "Clustering Groups",
        df["Cluster"].nunique()
    )

    st.divider()

    st.header("Problem Statement")

    st.write(
        "Finding suitable books from a large collection can be "
        "difficult for readers. This project develops an intelligent "
        "recommendation system that suggests books based on textual "
        "similarities, genres and clustering techniques."
    )

    st.header("Project Objectives")

    st.write("""
    1. Collect and combine book datasets.
    2. Perform data cleaning and preprocessing.
    3. Analyze book information using EDA.
    4. Apply Natural Language Processing techniques.
    5. Convert book information into numerical features using TF-IDF.
    6. Group similar books using K-Means clustering.
    7. Develop content-based, clustering-based and hybrid models.
    8. Evaluate recommendation performance.
    9. Develop an interactive Streamlit application.
    """)

    st.header("Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        - Python
        - Pandas and NumPy
        - Matplotlib and Seaborn
        - Natural Language Processing
        - TF-IDF Vectorization
        """)

    with col2:
        st.write("""
        - Cosine Similarity
        - K-Means Clustering
        - Scikit-learn
        - Joblib
        - Streamlit
        """)

    st.header("Recommendation Techniques")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Content-Based")
        st.write(
            "Finds books with similar textual features "
            "using TF-IDF and cosine similarity."
        )

    with col2:
        st.subheader("Clustering-Based")
        st.write(
            "Recommends books belonging to the same "
            "K-Means cluster."
        )

    with col3:
        st.subheader("Hybrid")
        st.write(
            "Combines textual similarity and cluster "
            "membership."
        )

    st.divider()

    st.header("Project Workflow")

    st.info(
        "Data Collection → Data Cleaning → EDA → NLP → "
        "TF-IDF → K-Means → Recommendation Models → "
        "Evaluation → Streamlit"
    )


def recommendation_page():

    st.title("Book Recommendations")

    st.write(
        "Discover books based on your favourite genre "
        "or a book you already like."
    )

    st.divider()

    st.header("1. Genre-Based Recommendations")

    st.write(
        "Choose a genre to discover highly rated books "
        "from the dataset."
    )

    genre = st.selectbox(
        "Select Your Favourite Genre",
        ["Science Fiction", "Thriller"]
    )

    if st.button("Find Books by Genre", use_container_width=True):

        genre_results = get_genre_recommendations(
            genre,
            top_n=5
        )

        st.subheader(f"Top 5 {genre} Books")

        if genre_results.empty:
            st.warning("No books found for this genre.")

        else:
            st.dataframe(
                genre_results,
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    st.header("2. Similar Book Recommendations")

    st.write(
        "Select a book and choose a recommendation method "
        "to find similar books."
    )

    method = st.selectbox(
        "Choose Recommendation Method",
        [
            "Content-Based",
            "Clustering-Based",
            "Hybrid"
        ]
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    book_options = (
        df[["Book Name", "Author"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Book Name")
    )

    book_choices = [
        (row["Book Name"], row["Author"])
        for _, row in book_options.iterrows()
    ]

    selected_book = st.selectbox(
        "Search and Select a Book",
        book_choices,
        format_func=lambda book: f"{book[0]} — {book[1]}"
    )

    if st.button(
        "Recommend Similar Books",
        use_container_width=True
    ):

        matching_rows = df.index[
            (df["Book Name"] == selected_book[0])
            & (df["Author"] == selected_book[1])
        ].tolist()

        if not matching_rows:

            st.warning("Selected book was not found.")

        else:

            book_index = matching_rows[0]

            book = df.loc[book_index]

            st.subheader("Selected Book")

            col1, col2, col3 = st.columns(3)

            col1.write("**Book Title**")
            col1.write(book["Book Name"])

            col2.write("**Author**")
            col2.write(book["Author"])

            col3.write("**Rating**")

            if pd.notna(book["Rating"]):
                col3.write(f"{book['Rating']:.2f} / 5")
            else:
                col3.write("Not available")

            recommendations = get_recommendations(
                book_index,
                method,
                top_n
            )

            st.subheader("Recommended Books")

            if recommendations.empty:

                st.warning("No recommendations available.")

            else:

                recommendations["Recommendation Score"] = (
                    recommendations["Recommendation Score"].round(4)
                )

                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Recommendation scores are ranking scores, "
                    "not probabilities or model accuracy."
                )


def show_chart(filename, title):

    chart_path = FIGURES_DIR / filename

    st.subheader(title)

    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)

    else:
        st.warning(f"Chart not found: {filename}")


def eda_dashboard():

    st.title("Exploratory Data Analysis Dashboard")

    st.write(
        "Explore important insights and patterns "
        "identified in the book dataset."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Books", f"{len(df):,}")

    col2.metric(
        "Average Rating",
        f"{df['Rating'].mean():.2f} / 5"
    )

    col3.metric(
        "Total Genres Available",
        f"{sum(df['Ranks and Genre'].apply(lambda x: len(extract_genres(x)))):,}"
    )

    st.divider()

    st.header("Book Analysis")

    col1, col2 = st.columns(2)

    with col1:
        show_chart(
            "rating_distribution.png",
            "Rating Distribution"
        )

    with col2:
        show_chart(
            "price_distribution.png",
            "Price Distribution"
        )

    col1, col2 = st.columns(2)

    with col1:
        show_chart(
            "reviews_distribution.png",
            "Reviews Distribution"
        )

    with col2:
        show_chart(
            "listening_time_distribution.png",
            "Listening Time Distribution"
        )

    st.divider()

    st.header("Genre Analysis")

    col1, col2 = st.columns(2)

    with col1:
        show_chart(
            "popular_genres.png",
            "Most Popular Genres"
        )

    with col2:
        show_chart(
            "rating_by_genre.png",
            "Ratings Across Genres"
        )

    st.divider()

    st.header("Author Analysis")

    col1, col2 = st.columns(2)

    with col1:
        show_chart(
            "top_authors_by_rating.png",
            "Top Authors by Average Rating"
        )

    with col2:
        show_chart(
            "author_popularity_vs_rating.png",
            "Author Popularity vs Rating"
        )

    st.divider()

    st.header("Correlation Analysis")

    show_chart(
        "correlation_heatmap.png",
        "Correlation Heatmap"
    )

    st.divider()

    st.header("K-Means Cluster Analysis")

    show_chart(
        "cluster_distribution.png",
        "Books in Each Cluster"
    )

    if "Cluster" in df.columns:

        cluster_counts = (
            df["Cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Number of Books")
        )

        st.dataframe(
            cluster_counts,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.header("Hidden Gems")

    st.write(
        "Highly rated books with relatively few reviews."
    )

    if HIDDEN_GEMS_PATH.exists():

        hidden_gems = pd.read_csv(HIDDEN_GEMS_PATH)

        st.dataframe(
            hidden_gems[
                ["Book Name", "Author", "Rating", "Number of Reviews"]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Hidden gems file not found.")


st.sidebar.title("Audible Insights")

st.sidebar.write("Intelligent Book Recommendation System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Book Recommendations",
        "EDA Dashboard"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Developed using Python, NLP and Machine Learning"
)

if page == "Home":

    introduction_page()

elif page == "Book Recommendations":

    recommendation_page()

else:

    eda_dashboard()