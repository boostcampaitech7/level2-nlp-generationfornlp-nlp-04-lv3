import pdfplumber
import re
import json
import unicodedata
from collections import deque


class ExtractGrade7Exam:

    def __init__(self, file_path):
        self.file_path = file_path
        self.q_start_pattern = r"^(문 (\d{2}|\d{1})\.\s|(\d{2}|\d{1})\.\s)"

    def run(self):
        question_list = list()

        # 문제 추출
        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                width, height = page.width, page.height
                left_layout = (0, 0, width / 2, height)
                right_layout = (width / 2, 0, width, height)

                left_lines = page.within_bbox(left_layout).extract_text_lines()
                right_lines = page.within_bbox(right_layout).extract_text_lines()

                left_images = page.within_bbox(left_layout).images
                right_images = page.within_bbox(right_layout).images

                left_q_list = self.extract_question(deque(left_lines), left_images)
                right_q_list = self.extract_question(deque(right_lines), right_images)
                question_list.extend(left_q_list)
                question_list.extend(right_q_list)

        # 문제 파일로 저장
        self.save_question(question_list)

    def extract_question(self, page, images):
        question_list = []

        # 해당 페이지의 첫 문제 시작 줄 찾기
        while page:
            line = page[0]
            if re.match(self.q_start_pattern, line["text"]):
                break
            page.popleft()

        # 문제 추출
        while page:
            line = page.popleft()
            match = re.match(self.q_start_pattern, line["text"])
            question_id = re.sub(f"[^0-9]", "", line["text"][: match.end()])

            # 1. 질문 추출
            question = [line]
            while page:
                # 1. 질문 ? (추가 설명) -> 질문 끝
                if re.search(r"\? \(.*\)$", line["text"]):
                    break

                # 2. 질문 ? -> 다음 줄에 추가 설명 올 수 있음
                if re.search(r"\?$", line["text"]):
                    while page:
                        if re.search(r"(^\(가\)|^\( ㉠ \))", page[0]["text"]):
                            break
                        if not re.search(r"^\(.*", page[0]["text"]):
                            break
                        question.append(page.popleft())
                    break

                # 3. 질문 ? (추가 설명 -> 다음 줄에 추가 설명이 이어짐
                if re.search(r"\? \(.*$", line["text"]):
                    while page:
                        line = page.popleft()
                        question.append(line)
                        if re.search(r"\)$", line["text"]):
                            break
                    break

                line = page.popleft()
                question.append(line)

            # 보기 추출
            context = []
            while page:
                if re.search("①", page[0]["text"]):
                    break
                context.append(page.popleft())

            # 선지 찾기
            choices = []
            while page:
                if re.match(self.q_start_pattern, page[0]["text"]):
                    break
                choices.append(page.popleft())

            # 문제에 그림 있을 경우 배제
            if self.is_with_img(question, choices, images):
                continue
            # 추출한 데이터(지문, 보기, 선지) 저장
            questionset = dict()
            questionset["question_id"] = question_id
            questionset["question"] = re.sub(
                self.q_start_pattern, "", " ".join([q["text"] for q in question])
            )
            questionset["context"] = " ".join([c["text"] for c in context])
            questionset["choices"] = [
                choice.strip()
                for choice in re.split(
                    r"(①|②|③|④|⑤)", " ".join([choice["text"] for choice in choices])
                )
            ][::2][1:]
            question_list.append(questionset)

        return question_list

    def save_question(self, question_list):
        save_file_name = f"{self.file_path.split('/')[-1].split('.')[0]}"

        with open(f"{save_file_name}.json", "w", encoding="utf-8") as json_file:
            json.dump(question_list, json_file, ensure_ascii=False, indent=4)

    def is_with_img(self, question, choices, images):
        top = question[0]["top"]
        bottom = choices[-1]["bottom"]

        for img in images:
            if top < img["top"] < bottom:
                return True
        return False


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
        # "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2020_심리학.pdf",
        # "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2021_심리학.pdf",
        # "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2022_심리학.pdf",
        # "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2023_심리학.pdf",
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/psychology/7급공채_2024_심리학.pdf",
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
