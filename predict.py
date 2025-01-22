import argparse
from pyexpat import model


def predict_csv(csv_path, model_name, output_csv_path="", text_label="sample_text", build_prompt=True):
    df = pd.read_csv(csv_path)
    predict(df, model_name, output_csv_path, text_label, build_prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("-p", "csv_path", required=True)
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("-o", "--output_csv_path")
    parser.add_argument("-t", "--text_label")
    args = parser.parse_args()

    print(args)

    predict_csv(args.csv_path, args.model_name, args)
