import pandas as pd
import os
from delete_circle_number import process_military_korean_problems


def process_csv():
    # Read the CSV file
    input_file = "/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military-korean.csv"
    output_file = "/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military-korean_processed.csv"

    df = pd.read_csv(input_file)

    # Process the problems column
    df["problems"] = df["problems"].apply(process_military_korean_problems)

    # Save the processed data
    df.to_csv(output_file, index=False)
    print(f"Processed file saved to: {output_file}")


if __name__ == "__main__":
    process_csv()
