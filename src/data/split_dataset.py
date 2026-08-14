import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    csv_path,
    train_path,
    test_path,
    test_size=0.2,
    random_state=42
):

    df = pd.read_csv(csv_path)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        random_state=random_state
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print("=" * 50)
    print("Dataset Split Completed")
    print("=" * 50)
    print(f"Train Samples : {len(train_df)}")
    print(f"Test Samples  : {len(test_df)}")