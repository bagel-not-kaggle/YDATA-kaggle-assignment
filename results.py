import argparse


def results(csv_path, per_dataset):
    predictions_df = pd.read_csv(csv_path)
    if per_dataset:
        datasets = predictions_df["dataset_name"].unique().tolist()
        for dataset in datasets:
            df = predictions_df[predictions_df["dataset_name"] ==dataset]
            calculate_classification_report(df)
    else:
        calculate_classification_report(predictions_df)

def calculate_classification_report():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Results")
    parser.add_argument("-p", "csv_path", default=DEFAULT_PREDICTION_CSV)
    parser.add_argument("-pd", "--per_dataset", action="store_true", default=False)
    args = parser.parse_args()

    results = results(args.csv_path, args.per_dataset)
