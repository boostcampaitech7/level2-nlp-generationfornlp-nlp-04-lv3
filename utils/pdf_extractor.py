import pdfplumber
import re
import csv

file_name = "2025_ethics"


def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 페이지 너비와 높이 가져오기
            width = page.width
            height = page.height

            # 왼쪽 컬럼 추출
            left_bbox = (0, 0, width / 2, height)
            left_crop = page.crop(left_bbox)
            left_text = left_crop.extract_text() or ""

            # 오른쪽 컬럼 추출
            right_bbox = (width / 2, 0, width, height)
            right_crop = page.crop(right_bbox)
            right_text = right_crop.extract_text() or ""

            # 두 컬럼의 텍스트 합치기
            page_text = left_text + "\n" + right_text
            full_text += page_text + "\n"
    return full_text


def parse_text_to_data(text):
    data = []
    # 문제 세트 구분 패턴
    problem_set_pattern = re.compile(
        r"\[(\d+～\d+)\] 다음 글을 읽고 물음에 답하시오\.(.*?)\n(?=\d+\.)", re.DOTALL
    )
    # 개별 문제 구분 패턴
    question_pattern = re.compile(
        r"(\d+)\.\s*(.*?)\n(.*?)(①.*?⑤.*?)(?=\n\d+\.|\Z)", re.DOTALL
    )

    # 문제 세트 추출
    problem_sets = problem_set_pattern.finditer(text)
    id_counter = 0
    for problem_set in problem_sets:
        problem_numbers = problem_set.group(1)
        paragraph = problem_set.group(2).strip().replace("\n", " ").replace("\r", " ")
        paragraph = re.sub(r"\d*\s*홀수형", "", paragraph)
        start_index = problem_set.end()

        # 다음 문제 세트나 문서의 끝까지를 범위로 설정
        next_problem_set = problem_set_pattern.search(text, pos=problem_set.end())
        if next_problem_set:
            end_index = next_problem_set.start()
        else:
            end_index = len(text)
        problem_text = text[start_index:end_index]

        # 개별 문제 추출
        questions = question_pattern.finditer(problem_text)
        for question in questions:
            id_counter += 1
            q_number = question.group(1)
            q_content_full = question.group(2).strip()
            between_text = question.group(3).strip()
            choices_text = question.group(4).strip()

            # <보기> 내용 추출
            question_plus = ""
            if "<보기>" in between_text:
                # <보기>부터 선택지 앞까지의 내용 추출
                question_plus_pattern = re.compile(r"<보기>(.*?)$", re.DOTALL)
                match = question_plus_pattern.search(between_text)
                if match:
                    question_plus = match.group(1).strip()
            else:
                # <보기>가 없으면 question_plus는 빈 문자열
                question_plus = ""

            # 문제 내용은 <보기> 이전의 between_text와 q_content_full 합침
            if "<보기>" in between_text:
                q_content = (
                    q_content_full + "\n" + between_text.split("<보기>")[0].strip()
                )
            else:
                q_content = q_content_full + "\n" + between_text

            # 선택지 정리
            choice_list = re.findall(
                r"①\s*(.*?)\s*②\s*(.*?)\s*③\s*(.*?)\s*④\s*(.*?)\s*⑤\s*(.*)",
                choices_text,
                re.DOTALL,
            )
            if choice_list:
                choices = [
                    choice_list[0][0],
                    choice_list[0][1],
                    choice_list[0][2],
                    choice_list[0][3],
                    choice_list[0][4],
                ]
            else:
                choices = []
            patterns_to_remove = [
                r"\n\d+",  # "\n숫자" 패턴
                r"\n수학능력시험 문제지",  # "\n수학능력시험 문제지"
                r"\n홀수형",  # "\n홀수형"
                r"\n짝수형",  # "\n짝수형"
            ]

            def clean_choice(choice, patterns):
                for pattern in patterns:
                    choice = re.sub(pattern, "", choice)
                return choice.strip()

            choices = [clean_choice(choice, patterns_to_remove) for choice in choices]
            choices = [choice.replace("\n", " ") for choice in choices]
            q_content = q_content.strip().replace("\n", " ")
            # <보 기> 이후 내용 제거 및 제거된 내용 저장
            match = re.search(r"<보 기>(.*)", q_content, re.DOTALL)
            if match:
                question_plus = match.group(1).strip()  # <보 기> 뒷부분 내용 저장
                q_content = re.sub(
                    r"<보 기>.*", "<보 기>", q_content, flags=re.DOTALL
                )  # <보 기> 이후 내용 제거
            else:
                question_plus = ""
            q_content = q_content.replace("[3점]", "").replace("<보 기>", "").strip()
            # problems 필드에 문제 내용과 선택지를 저장
            problems_dict = {
                "question": q_content,
                "choices": choices,
                "answer": "",
            }

            data.append(
                {
                    "id": f"generation-for-nlp-{id_counter}",
                    "paragraph": paragraph,
                    "problems": str(problems_dict),
                    "question_plus": question_plus,
                }
            )
    return data


def save_data_to_csv(data, csv_file_path):
    fieldnames = ["id", "paragraph", "problems", "question_plus"]
    with open(csv_file_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow(item)


# 사용 예시
pdf_path = f"/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/pdf/{file_name}.pdf"  # PDF 파일 경로를 입력하세요
text = extract_text_from_pdf(pdf_path)
data = parse_text_to_data(text)
csv_file_path = f"./validation_output/{file_name}_pdf_output.csv"
save_data_to_csv(data, csv_file_path)
print(f"Data has been saved to {csv_file_path}")
