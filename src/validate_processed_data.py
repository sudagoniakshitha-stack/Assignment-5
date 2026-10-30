from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent.parent

file_path = (
    project_dir
    / "data"
    / "processed"
    / "cleaned_books.csv"
)

if not file_path.exists():
    print("ERROR: Cleaned dataset was not found.")
else:
    df = pd.read_csv(file_path)

    print("CLEANED DATASET VALIDATION")
    print("Shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate title-author combinations:")
    print(
        df.duplicated(
            subset=["Book Name", "Author"]
        ).sum()
    )

    print("\nRating statistics:")
    print(df["Rating"].describe())

    print("\nValidation completed.")
    print("\nLISTENING TIME CHECK")

    listening = df["Listening Time"].dropna()

    print("Total available values:", len(listening))
    print("Zero-minute values:", (listening == 0).sum())
    print("Negative values:", (listening < 0).sum())

    print("\nSample listening times:")
    print(listening.head(20).tolist())