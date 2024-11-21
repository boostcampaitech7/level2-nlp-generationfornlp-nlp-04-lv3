import pandas as pd
import os

# Validation output folder where the files are located
folder_path = (
    "/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/validation_output"
)

# Read all CSV files in the folder and combine them into one DataFrame
all_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
combined_data = pd.DataFrame()

for file in all_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    combined_data = pd.concat([combined_data, df], ignore_index=True)

# Save the combined DataFrame to a single CSV file
output_path = "/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/2025_KSAT_concat.csv"
combined_data.to_csv(output_path, index=False)
