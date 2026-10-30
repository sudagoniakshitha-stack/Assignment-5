import pandas as pd

# Load the original datasets
catalog = pd.read_csv("data/raw/Audible_Catlog.csv")

advanced = pd.read_csv(
    "data/raw/Audible_Catlog_Advanced_Features.csv"
)

# Display dataset information
print("DATASET 1")
print("Shape:", catalog.shape)
print("\nColumns:")
print(catalog.columns.tolist())
print("\nMissing values:")
print(catalog.isnull().sum())

print("\n" + "=" * 50)

print("DATASET 2")
print("Shape:", advanced.shape)
print("\nColumns:")
print(advanced.columns.tolist())
print("\nMissing values:")
print(advanced.isnull().sum())

# Additional data quality checks

print("\n" + "=" * 60)
print("ADDITIONAL DATA QUALITY CHECKS")
print("=" * 60)

# Check duplicate rows
print("\nDataset 1 exact duplicate rows:")
print(catalog.duplicated().sum())

print("\nDataset 2 exact duplicate rows:")
print(advanced.duplicated().sum())

# Check rating ranges and unusual values
print("\nDataset 1 rating statistics:")
print(catalog["Rating"].describe())

print("\nDataset 1 unique ratings:")
print(sorted(catalog["Rating"].dropna().unique()))

print("\nDataset 2 rating statistics:")
print(advanced["Rating"].describe())

print("\nDataset 2 unique ratings:")
print(sorted(advanced["Rating"].dropna().unique()))

# Check missing values in all columns
print("\nDataset 1 missing values:")
print(catalog.isnull().sum())

print("\nDataset 2 missing values:")
print(advanced.isnull().sum())

# Check data types
print("\nDataset 1 data types:")
print(catalog.dtypes)

print("\nDataset 2 data types:")
print(advanced.dtypes)

# Check repeated book titles and authors

print("\n" + "=" * 60)
print("REPEATED BOOKS AND AUTHORS")
print("=" * 60)

# Dataset 1
print("\nDataset 1 repeated title-author combinations:")

duplicates_1 = catalog.duplicated(
    subset=["Book Name", "Author"],
    keep=False
)

print("Number of rows involved:", duplicates_1.sum())

print(
    catalog.loc[
        duplicates_1,
        ["Book Name", "Author", "Rating", "Number of Reviews", "Price"]
    ].head(10)
)

# Dataset 2
print("\nDataset 2 repeated title-author combinations:")

duplicates_2 = advanced.duplicated(
    subset=["Book Name", "Author"],
    keep=False
)

print("Number of rows involved:", duplicates_2.sum())

print(
    advanced.loc[
        duplicates_2,
        ["Book Name", "Author", "Rating", "Number of Reviews", "Price"]
    ].head(10)
)
# Investigate conflicting information for repeated books

def inspect_repeated_books(df, dataset_name):
    print(f"\n{'=' * 60}")
    print(f"{dataset_name}: CONFLICT CHECK")
    print("=" * 60)

    key = ["Book Name", "Author"]

    repeated = df[
        df.duplicated(subset=key, keep=False)
    ].copy()

    # Count distinct values for each book-author combination
    grouped = repeated.groupby(key, dropna=False).agg(
        rating_values=("Rating", "nunique"),
        review_values=("Number of Reviews", "nunique"),
        price_values=("Price", "nunique"),
        row_count=("Rating", "size")
    ).reset_index()

    print("\nRepeated combinations:", len(grouped))

    print(
        "\nCombinations with different ratings:",
        (grouped["rating_values"] > 1).sum()
    )

    print(
        "Combinations with different review counts:",
        (grouped["review_values"] > 1).sum()
    )

    print(
        "Combinations with different prices:",
        (grouped["price_values"] > 1).sum()
    )

    print("\nFirst 10 combinations to investigate:")
    print(grouped.head(10).to_string(index=False))


inspect_repeated_books(catalog, "DATASET 1")
inspect_repeated_books(advanced, "DATASET 2")
# Check which columns differ for repeated books

def check_all_column_conflicts(df, dataset_name):

    print("\n" + "=" * 60)
    print(dataset_name)
    print("ALL-COLUMN CONFLICT CHECK")
    print("=" * 60)

    key = ["Book Name", "Author"]

    repeated = df[
        df.duplicated(subset=key, keep=False)
    ].copy()

    # Count distinct values in every column for each book-author group
    grouped = repeated.groupby(
        key,
        dropna=False
    ).nunique(dropna=False)

    print("\nNumber of repeated title-author groups:")
    print(len(grouped))

    print("\nGroups with different values in each column:")

    for column in df.columns:
        if column not in key:
            count = (grouped[column] > 1).sum()

            print(f"{column}: {count}")

    print("\nFirst 5 groups with any conflicting values:")

    conflict_mask = grouped.drop(
        columns=key,
        errors="ignore"
    ).gt(1).any(axis=1)

    conflicting_groups = grouped[conflict_mask]

    print(conflicting_groups.head(5))


check_all_column_conflicts(
    catalog,
    "DATASET 1"
)

check_all_column_conflicts(
    advanced,
    "DATASET 2"
)
print("\nLISTENING TIME INVESTIGATION")

advanced = pd.read_csv(
    "data/raw/Audible_Catlog_Advanced_Features.csv"
)

print("\nOriginal columns:")
print(advanced.columns.tolist())

if "Listening Time" in advanced.columns:
    print("\nOriginal listening time values:")
    print(advanced["Listening Time"].head(10))

    print("\nMissing listening times:")
    print(advanced["Listening Time"].isna().sum())

    print("\nUnique listening time values:")
    print(advanced["Listening Time"].unique()[:20])

from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent

file_path = (
    project_dir
    / "data"
    / "raw"
    / "Audible_Catlog_Advanced_Features.csv"
)

df = pd.read_csv(file_path)

print("\nLISTENING TIME FORMAT CHECK")

print("\nFirst 30 original values:")
print(df["Listening Time"].head(30).to_string(index=False))

print("\nValues containing only minutes:")
print(
    df[
        df["Listening Time"].astype(str).str.contains(
            r"^\s*\d+\s+minutes?\s*$",
            case=False,
            na=False
        )
    ]["Listening Time"].head(20).to_string(index=False)
)
print("\nValues containing only hours:")
print(
    df.loc[
        df["Listening Time"].astype(str).str.contains(
            r"^\s*\d+\s+hours?\s*$",
            case=False,
            na=False
        ),
        "Listening Time"
    ].head(20).to_string(index=False)
)

print("\nValues containing only minutes:")
print(
    df.loc[
        df["Listening Time"].astype(str).str.contains(
            r"^\s*\d+\s+minutes?\s*$",
            case=False,
            na=False
        ),
        "Listening Time"
    ].head(20).to_string(index=False)
)
