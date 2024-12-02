import pandas as pd
import re
import fitz  # PyMuPDF

# 시험명 설정 (변경 가능) 예시: {시험명}-{과목}-{연도}-{인덱스}
exam = "qualification"  # 시험명
subject = "history_of_Korea"  # 과목
year = "2022-1"  # 연도
exam_index = 5  # 국어: 0(language), 사회: 3(society), 한국사: 5(history_of_Korea)

problem_pdf_path = "/data/ephemeral/home/ingue/level2-nlp-generationfornlp-nlp-04-lv3/utils/old_exam_data/검정고시_2024-2_사회.pdf"  # 문제 경로
answer_pdf_path = "/data/ephemeral/home/ingue/level2-nlp-generationfornlp-nlp-04-lv3/utils/old_exam_data/2024년도_제2회_고졸_정답(확정안).pdf"
csv_output_path = f"{exam}-{subject}-{year}.csv"  # 변경 가능


def extract_problems_from_pdf(pdf_path):
    # PDF 열기
    doc = fitz.open(pdf_path)
    questions = []
    question_id = 1  # 문제 ID 초기값

    # 제외할 패턴 정의(검정고시)
    EXCLUDE_PATTERNS = [
        r"\d{4}년도 제\d회 고등학교 졸업학력 검정고시",  # '2024년도 제2회 ...' 형태
        r"고졸",  # '고졸'
        r"제\d 교시",  # '제4 교시'
        r"사\s{2,}회|국\s{2,}어|수\s{2,}학|과\s{2,}학|한\s{2,}국\s{2,}사"  # '사    회', '국    어' 등 과목명
        r"\s*\((사회|한국사|국어|수학|과학|도덕|영어|기술|가정|체육|음악|미술)\)\s*\d+－\d+"  # '(사회)  2－1', '(국어)  2－1' 등 제거
        r"\d{4}년 제\d+회 공인중개사 \d차 \d교시 B형-\d+-\d+",  # '2020년 제31회 공인중개사 2차 1교시 B형-12-2'와 같은 형식
    ]

    for page_number in range(len(doc)):
        page = doc[page_number]
        # 페이지 텍스트 추출
        text = page.get_text("text")

        for pattern in EXCLUDE_PATTERNS:
            text = re.sub(pattern, "", text)

        # 페이지 텍스트 출력 (디버깅용)
        print(f"--- Page {page_number + 1} ---\n{text}\n")

        # 문제 추출 (유연한 정규식)
        question_pattern = r"(\d+)\.\s(.+?)(?=(\d+\.\s)|$)"  # 문제 번호로 시작, 다음 문제 번호 또는 끝까지
        matches = re.findall(question_pattern, text, re.DOTALL)

        for match in matches:
            question_text = match[1].strip()

            # 질문과 선택지 분리
            question_parts = re.split(r"\n", question_text)

            # 반복 지문 확인
            repeated_match = re.search(
                r"\[\d+～\d+\](.+)물음에 답하시오\.", question_text
            )
            if repeated_match:
                common_paragraph = repeated_match.group(1).strip()

            indices0 = next(
                (index for index, item in enumerate(question_parts) if "것은?" in item),
                -1,
            )
            indices1 = next(
                (index for index, item in enumerate(question_parts) if "①" in item), -1
            )
            indices2 = next(
                (index for index, item in enumerate(question_parts) if "②" in item), -1
            )
            indices3 = next(
                (index for index, item in enumerate(question_parts) if "③" in item), -1
            )
            indices4 = next(
                (index for index, item in enumerate(question_parts) if "④" in item), -1
            )
            passage = question_parts[0].strip()
            question = {
                "question": passage,
                "choices": [
                    "".join(question_parts[indices1:indices2])[1:],
                    "".join(question_parts[indices2:indices3])[1:],
                    "".join(question_parts[indices3:indices4])[1:],
                    "".join(question_parts[indices4:])[1:],
                ],
            }

            questions.append(
                {
                    "id": f"{exam}-{subject}-{year}-{question_id}",
                    "paragraph": "".join(question_parts[1:indices1]),
                    "problems": question,  # JSON 형식으로 저장
                }
            )

            question_id += 1

    return questions


# 정답 추출 함수
def extract_answers_from_pdf(pdf_path):
    """PDF에서 정답 추출"""
    doc = fitz.open(pdf_path)
    answers = {}

    page = doc[exam_index]
    text = page.get_text("text")

    # 정답 추출
    answer_pattern = r"(\d+)\s+([①②③④])"
    matches = re.findall(answer_pattern, text)
    for match in matches:
        question_num, answer = match
        answers[int(question_num)] = answer

    return answers


def combine_problems_and_answers(problems, answers):
    """문제와 정답을 매핑하여 CSV 형식으로 변환"""
    combined = []
    special_num = {"①": 1, "②": 2, "③": 3, "④": 4}

    for problem in problems:
        question_id = problem["id"]
        paragraph = problem["paragraph"]
        problems_json = problem["problems"]
        problems_num = int(re.search(r"-(\d+)$", question_id).group(1))
        problems_json["answer"] = special_num[
            answers[problems_num]
        ]  # 문제지와 답지의 갯수가 서로 맞지 않으면 주석 가능
        combined.append(
            {
                "id": question_id,
                "paragraph": paragraph,
                "problems": problems_json,
                "question_plus": "",
            }
        )

    return combined


def write_to_csv(data, output_path):
    # CSV로 저장
    df = pd.DataFrame(data)  # pandas 활용
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"CSV 파일로 저장 완료: {output_path}")


# 실행
problems = extract_problems_from_pdf(problem_pdf_path)
answers = extract_answers_from_pdf(answer_pdf_path)
combined_data = combine_problems_and_answers(problems, answers)
write_to_csv(combined_data, csv_output_path)
