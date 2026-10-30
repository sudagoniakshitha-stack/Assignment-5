# Audible Insights – Intelligent Book Recommendation System

## 1. Project Overview

Audible Insights is an intelligent book recommendation system developed using Python, Natural Language Processing (NLP), and Machine Learning.

The project analyzes audiobook information and recommends books based on textual similarities, genres, and clustering techniques.

## 2. Problem Statement

Finding suitable books from a large collection can be difficult for readers. This project aims to simplify book discovery by providing personalized and similar-book recommendations.

## 3. Project Objectives

* Combine and clean audiobook datasets.
* Perform Exploratory Data Analysis (EDA).
* Extract textual features using TF-IDF.
* Group similar books using K-Means clustering.
* Develop content-based, clustering-based, and hybrid recommendation methods.
* Evaluate recommendation approaches.
* Build an interactive application using Streamlit.

## 4. Technologies Used

* Python
* Pandas and NumPy
* Matplotlib and Seaborn
* Natural Language Processing
* TF-IDF Vectorization
* Cosine Similarity
* K-Means Clustering
* Scikit-learn
* Joblib
* Streamlit

## 5. Recommendation Methods

### Content-Based Recommendation

Uses TF-IDF and cosine similarity to identify books with similar textual information.

### Clustering-Based Recommendation

Recommends books belonging to the same K-Means cluster.

### Hybrid Recommendation

Combines textual similarity and cluster membership to rank recommended books.

## 6. Application Features

* Book recommendations based on genre.
* Similar-book recommendations.
* Three recommendation methods.
* Exploratory Data Analysis dashboard.
* Genre and author analysis.
* Hidden-gem identification.
* K-Means cluster visualization.

## 7. Project Workflow

Data Collection → Data Cleaning → EDA → NLP → TF-IDF → K-Means Clustering → Recommendation Models → Model Evaluation → Streamlit Application

## 8. Project Structure

```text
Audible Insights/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   └── figures/
├── src/
├── app.py
├── requirements.txt
└── README.md

## 9. Installation

Clone the repository:


git clone YOUR_GITHUB_REPOSITORY_LINK
Navigate to the project directory:

cd "Audible Insights"

Install the required libraries:

pip install -r requirements.txt

## 10. Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

## 11. Model Evaluation

The recommendation approaches are evaluated using Precision@5 and Recall@5 with rating-based proxy relevance.

The evaluation results should be interpreted in the context of this proxy-based approach and the available dataset.

## 12. Conclusion

Audible Insights demonstrates how Natural Language Processing and Machine Learning can be used to analyze audiobook data and build an interactive book recommendation system.
