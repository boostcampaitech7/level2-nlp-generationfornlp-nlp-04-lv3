import pdfplumber
import re
import json
from PyPDF2 import PdfReader
import unicodedata


class ExtractGrade7Exam:

    def __init__(self, file_path):

        self.file_path = file_path

    def is_problem_start_point(self, text):
        # print(text)
        match = re.search(r"^문 (\d{2}|\d{1})\.", text)
        # match = re.search(r"^(\d{2}|\d{1})\.", text)

        if match:
            return (
                match.group().replace("문", "problem").replace(".", ""),
                text[match.end() :],
            )
        else:
            return None

    def run(
        self,
    ):
        result = list()

        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                width, height = page.width, page.height
                left_layout = (0, 0, width / 2, height)  # 상하로 범위 확장
                right_layout = (width / 2, 0, width, height)

                left_text = page.within_bbox(left_layout).extract_text()
                right_text = page.within_bbox(right_layout).extract_text()

                sub_result1 = self.extract_page(left_text)
                sub_result2 = self.extract_page(right_text)
                result.extend(sub_result1)
                result.extend(sub_result2)

        with open(
            f"{self.file_path.split('/')[-1].split('.')[0]}.json", "w", encoding="utf-8"
        ) as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

    def extract_page(self, page_text):
        page_text = page_text.split("\n")
        result_list = []

        index = 0
        # 문제 시작 위치 찾기
        while index < len(page_text):
            is_start_point = self.is_problem_start_point(page_text[index])

            if is_start_point is not None:
                break
            index += 1

        while index < len(page_text):
            question_id, q_start = self.is_problem_start_point(page_text[index])

            # 질문 찾기
            question = [q_start]
            while index < len(page_text):
                q_end_pattern1 = re.search(r"\? \(.*\)$", page_text[index])
                q_end_pattern4 = re.search(r"\?$", page_text[index])
                q_end_pattern2 = re.search(r"\? \(.*$", page_text[index])

                index += 1
                if q_end_pattern1 is not None:
                    break

                if q_end_pattern4 is not None:
                    if re.search(r"(^\(가\)|^\( ㉠ \))", page_text[index]):
                        break
                    if re.search(r"^\(.*", page_text[index]) is None:
                        break

                    question.append(page_text[index])
                    while not re.search(r"\)$", page_text[index]):
                        index += 1
                        question.append(page_text[index])
                    index += 1
                    break

                if q_end_pattern2 is not None:
                    while index < len(page_text):
                        question.append(page_text[index])

                        q_end_pattern3 = re.search(r"\)$", page_text[index])
                        index += 1
                        if q_end_pattern3 is not None:
                            break
                    break
                question.append(page_text[index])

            # 보기 찾기
            context = []
            while index < len(page_text):
                if re.search("①", page_text[index]) is not None:
                    break
                else:
                    context.append(page_text[index])
                    index += 1
                    continue

            # 선지 찾기
            print(page_text[index])
            choices = []
            while index < len(page_text):
                next_problem_start_point = self.is_problem_start_point(page_text[index])

                if next_problem_start_point is not None:
                    break

                choices.append(page_text[index])
                print(page_text[index])
                index += 1

            question = " ".join(question).strip()
            context = " ".join(context).strip() if len(context) != 0 else None
            choices = [
                choice.strip()
                for choice in re.split(r"(①|②|③|④)", " ".join(choices))[::2][1:]
            ]

            print(question_id)
            print("질문")
            print(question)
            print("보기")
            print(context)
            print("선지")
            print(choices)
            print("=" * 50)
            result = dict()
            result["question_id"] = question_id
            result["question"] = question
            result["context"] = context
            result["choices"] = choices
            result_list.append(result)

        return result_list


# 질문과 정답 매핑 함수
def map_question_answer(q_file, a_file):

    # 데이터 가져오기
    with open(q_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # 정답 찾기
    answer_list = []
    subject_name = q_file.split(".")[0].split("_")[-1]
    with pdfplumber.open(a_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()

            if tables:
                for table in tables:
                    for row in table:
                        if unicodedata.normalize(
                            "NFC", row[1]
                        ) == unicodedata.normalize("NFC", subject_name):
                            answer_list = row[3:]
                            print(len(answer_list))
                            break

    for index, row in enumerate(data):
        answer = answer_list[int(row["question_id"].split(" ")[-1]) - 1]
        data[index]["answer"] = answer

    with open(
        f"{q_file.split('/')[-1].split('.')[0]}_정답포함.json", "w", encoding="utf-8"
    ) as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    file_path_list = [
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2020_심리학.pdf",
        # "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2021_심리학.pdf",
        # "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2022_심리학.pdf",
        # "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2023_심리학.pdf",
        # "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2024_심리학.pdf",
    ]
    for file_path in file_path_list:
        extract_data = ExtractGrade7Exam(file_path)
        extract_data.run()

    """
    # 질문과 정답 매핑
    political_path = [
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/political/7급공채_2020_국제정치학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/political/7급공채_2021_국제정치학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/political/7급공채_2022_국제정치학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/political/7급공채_2023_국제정치학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/political/7급공채_2024_국제정치학.json",
    ]
    psychology_path = [
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2020_심리학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2021_심리학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2022_심리학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2023_심리학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2024_심리학.json",
    ]
    economy_path = [
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/economy/7급공채_2020_경제학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/economy/7급공채_2021_경제학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/economy/7급공채_2022_경제학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/economy/7급공채_2023_경제학.json",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/economy/7급공채_2024_경제학.json",
    ]
    answer_path = [
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/answer/7급공채_2020_정답.pdf",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/answer/7급공채_2021_정답.pdf",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/answer/7급공채_2022_정답.pdf",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/answer/7급공채_2023_정답.pdf",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/answer/7급공채_2024_정답.pdf",
    ]

    for a, b, c, d in zip(political_path, psychology_path, economy_path, answer_path):
        map_question_answer(a, d)
        map_question_answer(b, d)
        map_question_answer(c, d)
    """
