import os
import re
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
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


if __name__ == "__main__":
    main()
