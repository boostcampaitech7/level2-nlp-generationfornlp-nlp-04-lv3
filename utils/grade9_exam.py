import os
import re
import ast
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pykospacing import Spacing


# 문제를 추출하는 함수
def extract_text(test_subject_kor, test_subject_eng, test_years):
    """
    주어진 과목과 연도에 대해 문제를 추출합니다.

    Args:
    - test_subject_kor (str): 과목명 (한글)
    - test_subject_eng (str): 과목명 (영문)
    - test_years (list): 처리할 연도 리스트

    Returns:
    - pd.DataFrame: 추출된 문제 데이터프레임
    """
    data = []

    for year in test_years:
        # 각 연도별 PDF 파일 경로 설정
        file_path = os.path.join(
            os.getenv("ROOT_DIR"),
            f"grade9_{test_subject_eng}/9급공채_{year}_{test_subject_kor}.pdf",
        )
        reader = PdfReader(file_path)
        id_counter = 0
        print(f"Processing year: {year} for subject: {test_subject_kor}")

        for page in reader.pages:
            text = page.extract_text()
            if text:
                # 문제 번호와 내용을 추출
                question_matches = re.finditer(
                    r"(문\s*\d+\.|\d+\.)\s*(.+?)(?=\s*①)", text, re.DOTALL
                )
                for match in question_matches:
                    question_id = (
                        f"grade9-public-{test_subject_eng}-{year}-{id_counter}"
                    )

                    # 문제 텍스트를 추출하고 '?' 기준으로 분리
                    full_question = match.group(2).strip()
                    question_split = re.split(r"\?", full_question, maxsplit=1)
                    question_text = (
                        question_split[0].strip() + "?"
                        if len(question_split) > 1
                        else full_question
                    )
                    paragraph_text = (
                        question_split[1].strip() if len(question_split) > 1 else ""
                    )

                    # 선택지(①, ②, ③, ④) 추출
                    choices_pattern = re.compile(
                        r"①\s*(.+?)\s*②\s*(.+?)\s*③\s*(.+?)\s*④\s*(.+?)(?=(문\s*\d+\.|\d+\.)|$)",
                        re.DOTALL,
                    )
                    choices_match = choices_pattern.search(text[match.end() :])
                    choices = (
                        [choices_match.group(i).strip() for i in range(1, 5)]
                        if choices_match
                        else []
                    )

                    # 추출된 데이터를 리스트에 추가
                    data.append(
                        {
                            "id": question_id,
                            "paragraph": paragraph_text,
                            "problems": {
                                "question": question_text,
                                "choices": choices,
                                "answer": 0,
                            },
                            "question_plus": None,
                        }
                    )
                    id_counter += 1

    # 데이터를 데이터프레임으로 변환 후 저장
    df = pd.DataFrame(data, columns=["id", "paragraph", "problems", "question_plus"])
    output_file = os.path.join(
        os.getenv("ROOT_DIR"), f"grade9-exam-data/9급공채_{test_subject_kor}.csv"
    )
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Extracted data saved to {output_file}")
    return df


# 문제 데이터의 띄어쓰기를 교정하는 함수
def correct_spacing_in_csv(test_subject_kor):
    """
    주어진 과목의 문제 데이터를 교정하여 저장합니다.
    """
    input_file = os.path.join(
        os.getenv("ROOT_DIR"), f"grade9-exam-data/9급공채_{test_subject_kor}.csv"
    )
    spacing = Spacing()

    df = pd.read_csv(input_file, encoding="utf-8")
    for column in df.columns:
        if column == "problems":
            # 'problems' 컬럼에 띄어쓰기 교정
            def correct_problems_spacing(problem_str):
                try:
                    problem_dict = ast.literal_eval(problem_str)
                    if "question" in problem_dict:
                        problem_dict["question"] = spacing(problem_dict["question"])
                    if "choices" in problem_dict:
                        problem_dict["choices"] = [
                            spacing(choice) for choice in problem_dict["choices"]
                        ]
                    return str(problem_dict)
                except Exception as e:
                    print(f"Error processing problems column: {e}")
                    return problem_str

            df["problems"] = df["problems"].astype(str).apply(correct_problems_spacing)
        else:
            # 다른 텍스트 컬럼에 PyKoSpacing 적용
            df[column] = (
                df[column]
                .astype(str)
                .apply(lambda x: spacing(x) if pd.notnull(x) else x)
            )
    # 교정된 데이터를 저장
    output_file = os.path.join(
        os.getenv("ROOT_DIR"), f"grade9-exam-data/9급공채_{test_subject_kor}_clean.csv"
    )
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"교정된 CSV 파일이 {output_file}에 저장되었습니다.")


# 정답 데이터를 업데이트하는 함수
def update_answers(subject_csv_path, answer_files, subject_name, output_csv):
    """
    각 연도의 답지 파일에서 특정 과목의 정답을 추출하고 문제 파일의 'problems' 컬럼에 정답을 업데이트합니다.
    """
    try:
        df = pd.read_csv(subject_csv_path, encoding="utf-8")
    except Exception as e:
        print(f"Error reading subject CSV file: {e}")
        return

    for year, answer_pdf_path in answer_files.items():
        try:
            reader = PdfReader(answer_pdf_path)
            text = "".join(page.extract_text() for page in reader.pages)

            # 2019년 특정 과목에서 책형 '나' 처리
            if year == "2019" and subject_name in ["국어", "한국사", "경제학개론"]:
                pattern = re.compile(rf"{subject_name}.*?나\s+([\d\s]+)", re.DOTALL)
            else:
                # 기본 정답 추출
                pattern = re.compile(rf"{subject_name}.*?\d+\s+([1-4\s]+)", re.DOTALL)

            match = pattern.search(text)
            if not match:
                print(f"Answers not found for {subject_name} in year {year}")
                continue

            answers = list(map(int, match.group(1).split()))
            if len(answers) != 20:
                print(
                    f"Unexpected number of answers for {subject_name} in year {year}: {len(answers)}"
                )
                continue

            # 연도별 데이터 시작 인덱스 계산
            start_index = sum(20 for y in answer_files.keys() if int(y) < int(year))
            for i in range(20):
                row_index = start_index + i
                if row_index >= len(df):
                    print(f"Row index {row_index} out of range for {year}")
                    break
                try:
                    # 정답 데이터를 업데이트
                    problems_dict = ast.literal_eval(df.at[row_index, "problems"])
                    problems_dict["answer"] = answers[i]
                    df.at[row_index, "problems"] = str(problems_dict)
                except Exception as e:
                    print(f"Error updating row {row_index} for year {year}: {e}")
                    continue

        except Exception as e:
            print(f"Error processing year {year} for {subject_name}: {e}")

    # 업데이트된 데이터 저장
    try:
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"Updated subject CSV saved to: {output_csv}")
    except Exception as e:
        print(f"Error saving updated subject CSV: {e}")


# 메인 실행 함수
def main():
    load_dotenv()

    # 과목 리스트 정의
    subjects = [
        {"kor": "경제학개론", "eng": "economy"},
        {"kor": "국어", "eng": "korean"},
        {"kor": "한국사", "eng": "history"},
    ]

    # 연도 리스트 정의
    years = ["2019", "2020", "2021", "2023", "2024"]

    for subject in subjects:
        # 문제 추출 및 저장
        extract_text(subject["kor"], subject["eng"], years)
        # 추출된 문제 데이터 띄어쓰기 교정
        correct_spacing_in_csv(subject["kor"])

    # 답지 경로 딕셔너리 정의
    answer_files = {
        "2019": os.path.join(os.getenv("ROOT_DIR"), "answer/9급공채_2019_답지.pdf"),
        "2020": os.path.join(os.getenv("ROOT_DIR"), "answer/9급공채_2020_답지.pdf"),
        "2021": os.path.join(os.getenv("ROOT_DIR"), "answer/9급공채_2021_답지.pdf"),
        "2023": os.path.join(os.getenv("ROOT_DIR"), "answer/9급공채_2023_답지.pdf"),
        "2024": os.path.join(os.getenv("ROOT_DIR"), "answer/9급공채_2024_답지.pdf"),
    }

    for subject in subjects:
        # 과목별 문제 파일 경로 불러오기
        input_csv = os.path.join(
            os.getenv("ROOT_DIR"),
            f"grade9-exam-data/9급공채_{subject['kor']}_clean.csv",
        )
        output_csv = os.path.join(
            os.getenv("ROOT_DIR"),
            f"grade9-exam-data/9급공채_{subject['kor']}_updated.csv",
        )
        # 띄어쓰기 교정한 문제 파일에 답 추가
        update_answers(input_csv, answer_files, subject["kor"], output_csv)


if __name__ == "__main__":
    main()
