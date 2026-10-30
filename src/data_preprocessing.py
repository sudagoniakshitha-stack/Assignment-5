from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

DATASET_1 = RAW_DIR / "Audible_Catlog.csv"
DATASET_2 = RAW_DIR / "Audible_Catlog_Advanced_Features.csv"

OUTPUT_FILE = PROCESSED_DIR / "cleaned_books.csv"


def load_datasets():
    if not DATASET_1.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_1}"
        )

    if not DATASET_2.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_2}"
        )

    catalog = pd.read_csv(DATASET_1)
    advanced = pd.read_csv(DATASET_2)

    print("\nSTEP 1: DATASETS LOADED")
    print("Catalog dataset shape:", catalog.shape)
    print("Advanced dataset shape:", advanced.shape)

    return catalog, advanced


def standardize_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    text_columns = [
        "Book Name",
        "Author",
        "Description",
        "Ranks and Genre"
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

            df[column] = df[column].replace(
                r"^\s*$",
                pd.NA,
                regex=True
            )

    numeric_columns = [
        "Rating",
        "Number of Reviews",
        "Price"       
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )
    if "Listening Time" in df.columns:
        listening_time = (
            df["Listening Time"]
            .astype("string")
            .str.lower()
            .str.strip()
        )
        hours = pd.to_numeric(
            listening_time.str.extract(
                r"(\d+)\s+hours?",
                expand = False
            ),
            errors="coerce"
        ).fillna(0)
        minutes = pd.to_numeric(
            listening_time.str.extract(
                r"(\d+)\s+minutes?",
                expand = False
            ),
            errors="coerce"
        ).fillna(0)
        df["Listening Time"] = (
            hours * 60+minutes
        )                
    for column in [
        "Rating",
        "Number of Reviews",
        "Price"      
    ]:
        if column in df.columns:
            df.loc[
                df[column] == -1,
                column
            ] = np.nan
    if "Rating" in df.columns:
        invalid_rating = (
            (df["Rating"] < 1)
            | (df["Rating"] > 5)
        )
        df.loc[
            invalid_rating,
            "Rating"
        ] = np.nan
    for column in [
        "Number of Reviews",
        "Price",
        "Listening Time"
    ]:
        if column in df.columns:
            df.loc[
                df[column] < 0,
                column
            ] = np.nan
    if "Description" in df.columns:
        description_text = (
            df["Description"]
            .astype("string")
            .str.lower()
        )
        error_patterns = (
            r"oops!.*rush hour",
            r"if you were trying to place an order",
            r"go to the amazon.*home page",
            r"conditions of use",
            r"privacy notice",
            r"try again in a short while"
        )

        invalid_description = description_text.str.contains(
            "|".join(error_patterns),
            regex=True,
            na=False
        )

        error_count = int(invalid_description.sum())

        if error_count > 0:
            print(
                "Invalid website descriptions removed:",
                error_count
            )

            df.loc[
                invalid_description,
                "Description"
            ] = pd.NA

    df["_book_key"] = (
        df["Book Name"]
        .astype("string")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["_author_key"] = (
        df["Author"]
        .astype("string")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


def clean_individual_dataset(df, dataset_name):
    print(f"\nSTEP 2: CLEANING {dataset_name}")

    original_rows = len(df)

    df = df.drop_duplicates().copy()

    print(
        "Exact duplicate rows removed:",
        original_rows - len(df)
    )

    df = standardize_data(df)

    before = len(df)

    df = df.dropna(
        subset=["Book Name", "Author"]
    ).copy()

    print(
        "Rows removed due to missing title/author:",
        before - len(df)
    )

    df = df[
        (df["_book_key"] != "")
        & (df["_author_key"] != "")
    ].copy()

    print("Rows after basic cleaning:", len(df))

    return df


def aggregate_books(df, dataset_name):
    print(f"\nSTEP 3: AGGREGATING {dataset_name}")

    original_rows = len(df)

    aggregation_rules = {}

    for column in [
        "Rating",
        "Number of Reviews",
        "Price",
        "Listening Time"
    ]:
        if column in df.columns:
            if column == "Rating":
                aggregation_rules[column] = "mean"
            elif column == "Number of Reviews":
                aggregation_rules[column] = "max"
            else:
                aggregation_rules[column] = "median"

    if "Description" in df.columns:
        def longest_text(values):
            valid_values = [
                str(value).strip()
                for value in values
                if pd.notna(value) and str(value).strip()
            ]

            if not valid_values:
                return np.nan

            return max(valid_values, key=len)

        aggregation_rules["Description"] = longest_text

    if "Ranks and Genre" in df.columns:
        def combine_text(values):
            valid_values = []

            for value in values:
                if pd.isna(value):
                    continue

                value = str(value).strip()

                if value and value != "-1":
                    if value not in valid_values:
                        valid_values.append(value)

            if not valid_values:
                return np.nan

            return " | ".join(valid_values)

        aggregation_rules["Ranks and Genre"] = combine_text

    grouped = df.groupby(
        ["_book_key", "_author_key"],
        as_index=False,
        dropna=False
    ).agg(aggregation_rules)

    display_names = (
        df.groupby(
            ["_book_key", "_author_key"],
            as_index=False
        )
        .agg({
            "Book Name": "first",
            "Author": "first"
        })
    )

    grouped = grouped.merge(
        display_names,
        on=["_book_key", "_author_key"],
        how="left",
        validate="one_to_one"
    )

    print("Rows before aggregation:", original_rows)
    print("Rows after aggregation:", len(grouped))

    return grouped


def merge_datasets(catalog, advanced):
    print("\nSTEP 4: MERGING DATASETS")

    catalog = catalog.rename(
        columns={
            "Book Name": "Book Name_catalog",
            "Author": "Author_catalog",
            "Rating": "Rating_1",
            "Number of Reviews": "Reviews_1",
            "Price": "Price_1"
        }
    )

    advanced = advanced.rename(
        columns={
            "Book Name": "Book Name_advanced",
            "Author": "Author_advanced",
            "Rating": "Rating_2",
            "Number of Reviews": "Reviews_2",
            "Price": "Price_2"
        }
    )

    merged = pd.merge(
        catalog,
        advanced,
        on=["_book_key", "_author_key"],
        how="outer",
        validate="one_to_one"
    )

    merged["Book Name"] = (
        merged["Book Name_catalog"]
        .combine_first(merged["Book Name_advanced"])
    )

    merged["Author"] = (
        merged["Author_catalog"]
        .combine_first(merged["Author_advanced"])
    )

    merged["Rating"] = merged[
        ["Rating_1", "Rating_2"]
    ].mean(axis=1)

    merged["Number of Reviews"] = merged[
        ["Reviews_1", "Reviews_2"]
    ].max(axis=1)

    merged["Price"] = merged[
        ["Price_1", "Price_2"]
    ].median(axis=1)

    if "Description" in merged.columns:
        merged["Description"] = merged[
            "Description"
        ].replace(r"^\s*$", np.nan, regex=True)

    if "Ranks and Genre" in merged.columns:
        merged["Ranks and Genre"] = merged[
            "Ranks and Genre"
        ].replace(r"^\s*$", np.nan, regex=True)

    merged = merged.drop(
        columns=["_book_key", "_author_key"],
        errors="ignore"
    )

    print("Merged dataset shape:", merged.shape)

    return merged


def save_processed_data(df):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSTEP 5: FINAL DATASET")
    print("Final dataset shape:", df.shape)

    duplicate_count = df.duplicated(
        subset=["Book Name", "Author"]
    ).sum()

    print(
        "Duplicate title-author combinations:",
        duplicate_count
    )

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    if "Rating" in df.columns:
        print("\nRating summary:")
        print(df["Rating"].describe())

    print("\nSaved processed dataset to:")
    print(OUTPUT_FILE)


def main():
    catalog, advanced = load_datasets()

    catalog = clean_individual_dataset(
        catalog,
        "CATALOG DATASET"
    )

    advanced = clean_individual_dataset(
        advanced,
        "ADVANCED FEATURES DATASET"
    )

    catalog = aggregate_books(
        catalog,
        "CATALOG DATASET"
    )

    advanced = aggregate_books(
        advanced,
        "ADVANCED FEATURES DATASET"
    )

    merged = merge_datasets(
        catalog,
        advanced
    )

    save_processed_data(merged)


if __name__ == "__main__":
    main()