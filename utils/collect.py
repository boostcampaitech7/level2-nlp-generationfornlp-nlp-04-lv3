import pandas as pd
import os


class MilitaryKoreanCSVMerger:
    def __init__(self, input_folder, output_file):
        self.input_folder = input_folder
        self.output_file = output_file

    def merge_csv_files(self):
        # 모든 CSV 파일의 경로를 가져옵니다.
        files = [f for f in os.listdir(self.input_folder) if f.endswith('.csv')]
        
        # 파일들을 읽어 DataFrame 목록에 저장합니다.
        df_list = []
        for file in files:
            file_path = os.path.join(self.input_folder, file)
            df = pd.read_csv(file_path)
            df_list.append(df)
        
        # 모든 DataFrame을 병합합니다.
        merged_df = pd.concat(df_list, ignore_index=True)
        # 병합된 DataFrame을 새로운 CSV 파일로 저장합니다.
        merged_df.to_csv(self.output_file, index=False)
        print(f"CSV 파일들이 병합되어 '{self.output_file}'로 저장되었습니다.")

def main():
    input_folder = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military'  # 병합할 CSV 파일들이 있는 폴더 경로
    output_file = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military/military_2025-2014_korean.csv'  # 병합된 결과를 저장할 파일 경로

    merger = MilitaryKoreanCSVMerger(input_folder, output_file)
    merger.merge_csv_files()


if __name__ == "__main__":
    main()