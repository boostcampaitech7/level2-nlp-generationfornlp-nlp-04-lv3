from datasets import load_dataset
import pandas as pd
import ast


def download_and_merge_kmmlu(output_file="KMMLU_korean_history_data.csv"):
    """
    Hugging Face에서 KMMLU 데이터셋의 모든 하위 데이터셋을 다운로드하고 병합합니다.
    병합된 데이터는 'KMMLU_merged_data.csv' 파일로 저장됩니다.
    """
    # 모든 subset 데이터를 담을 리스트
    all_data = []

    # KMMLU 데이터 로드
    subset_data = load_dataset("HAERAE-HUB/KMMLU", "Korean-History")

    # subset별 데이터프레임 생성 및 병합 리스트에 추가
    for subset_name in ["train", "dev", "test"]:
        df_subset = subset_data[subset_name].to_pandas()  # 데이터프레임으로 변환
        all_data.append(df_subset)

    # 모든 데이터를 하나의 DataFrame으로 병합
    merged_data = pd.concat(all_data, ignore_index=True)

    # 병합된 데이터 저장
    merged_data.to_csv(output_file, index=False)
    print(f"KMMLU 데이터셋 병합 완료 및 저장: '{output_file}'")


def extract_matching_data(original_data_path):
    """
    원본 데이터 파일에서 병합된 KMMLU 데이터셋에 포함된 데이터만 추출합니다.
    KMMLU 데이터의 'subset' 컬럼도 원본 데이터에 추가하여 저장합니다.

    Args:
        original_data_path (str): 원본 데이터 파일 경로 (CSV 형식이어야 함)
    """
    # 병합된 KMMLU 데이터셋 불러오기
    kmmlu_data = pd.read_csv("kmmlu_history.csv")

    # 원본 데이터 불러오기
    original_data = pd.read_csv(original_data_path)

    # 결과를 저장할 리스트 초기화
    extracted_data = []

    # KMMLU 데이터셋의 각 행을 순회하면서 원본 데이터와 조건에 맞는지 확인
    for _, kmmlu_row in kmmlu_data.iterrows():
        kmmlu_question = kmmlu_row["question"]
        kmmlu_choices = [kmmlu_row["A"], kmmlu_row["B"], kmmlu_row["C"], kmmlu_row["D"]]
        kmmlu_answer = kmmlu_row["answer"]

        # 원본 데이터에서 매칭되는 question과 choices 찾기
        for _, orig_row in original_data.iterrows():
            # problems 컬럼의 내용을 딕셔너리로 변환
            problem_data = ast.literal_eval(orig_row["problems"])
            original_question = problem_data["question"] + " " + orig_row["paragraph"]
            original_choices = problem_data["choices"]
            original_answer = problem_data["answer"]

            # 조건 1: question이 동일
            # 조건 2: choices 리스트가 동일
            if (
                (kmmlu_question == original_question)
                & (kmmlu_choices == original_choices)
                & (kmmlu_answer == original_answer)
            ):
                # 조건에 맞는 경우, 원본 데이터에 KMMLU의 subset 컬럼 추가
                matched_row = orig_row.copy()
                matched_row["Category"] = kmmlu_row["Category"]
                extracted_data.append(matched_row)

    # 추출된 데이터프레임 생성
    extracted_df = pd.DataFrame(extracted_data)

    # 추출된 데이터 저장
    extracted_df.to_csv("answer.csv", index=False)
    print(
        "원본 데이터에서 일치하는 데이터 추출 완료 및 저장: 'extracted_matching_data.csv'"
    )


if __name__ == "__main__":
    # KMMLU 데이터 병합
    download_and_merge_kmmlu()

    # # 원본 데이터에서 병합된 데이터와 일치하는 데이터만 추출
    # original_data_path = "/data/ephemeral/home/data/train.csv"  # 여기에 원본 데이터 경로 입력
    # extract_matching_data(original_data_path)
