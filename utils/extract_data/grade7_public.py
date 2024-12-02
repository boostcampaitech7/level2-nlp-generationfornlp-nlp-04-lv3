import pdfplumber
import re
import json
import unicodedata
from collections import deque
import pandas as pd
import os


class ExtractGrade7Exam:

    def __init__(self, question_file_path, answer_file_path=None):
        self.question_file_path = question_file_path
        self.answer_file_path = answer_file_path
        self.q_start_pattern = r"^(문 (\d{2}|\d{1})\.\s|(\d{2}|\d{1})\.\s)"

    def run(self, return_type="json"):
        question_list = list()

        # 문제 추출
        with pdfplumber.open(self.question_file_path) as pdf:
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

        # 문제와 정답 매핑
        if self.answer_file_path:
            question_list = self.map_qa(question_list)

        # 문제 파일로 저장
        if return_type == "json":
            self.save_by_json(question_list)
        elif return_type == "csv":
            self.save_by_csv(question_list)
        else:
            print(f"{return_type} 형식은 지원하지 않습니다.")

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
            questionset["question_id"] = self.rename_question_id(question_id)
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

    # 질문과 정답 매핑 함수
    def map_qa(self, question_list):

        # 정답 찾기
        answer_list = []
        subject_name = self.question_file_path.split(".")[0].split("_")[-1]
        with pdfplumber.open(self.answer_file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()

                if not tables:
                    continue
                for table in tables:
                    for row in table:
                        if unicodedata.normalize(
                            "NFC", row[1]
                        ) == unicodedata.normalize("NFC", subject_name):
                            answer_list = [re.sub(r"[^\d]", "", ele) for ele in row[3:]]
                            break

        for index, row in enumerate(question_list):
            answer = answer_list[int(row["question_id"].split("-")[-1]) - 1]
            question_list[index]["answer"] = int(answer)

        return question_list

    def save_by_json(self, question_list):
        save_file_name = f"{self.question_file_path.split('/')[-1].split('.')[0]}"

        with open(f"{save_file_name}.json", "w", encoding="utf-8") as json_file:
            json.dump(question_list, json_file, ensure_ascii=False, indent=4)

    def save_by_csv(self, question_list):
        save_file_name = f"{self.question_file_path.split('/')[-1].split('.')[0]}"

        id_list, paragraph_list, q_list = [], [], []
        for index, row in enumerate(question_list):
            id_list.append(row["question_id"])
            paragraph_list.append(row["context"])
            q_list.append(
                {
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": int(row["answer"]),
                }
            )
        df = pd.DataFrame()
        df["id"] = id_list
        df["paragraph"] = paragraph_list
        df["question"] = q_list
        df["question_plus"] = None
        df.to_csv(f"{save_file_name}.csv", index=False)

    def is_with_img(self, question, choices, images):
        top = question[0]["top"]
        bottom = choices[-1]["bottom"]

        for img in images:
            if top < img["top"] < bottom:
                return True
        return False

    def rename_question_id(self, id):
        subject_name = self.question_file_path.split(".")[0].split("_")[-1]
        year = self.question_file_path.split(".")[0].split("_")[-2]

        if unicodedata.normalize("NFC", subject_name) == unicodedata.normalize(
            "NFC", "심리학"
        ):
            subject_name = "psychology"
        elif unicodedata.normalize("NFC", subject_name) == unicodedata.normalize(
            "NFC", "국제정치학"
        ):
            subject_name = "politics"
        elif unicodedata.normalize("NFC", subject_name) == unicodedata.normalize(
            "NFC", "경제학"
        ):
            subject_name = "economics"

        return f"grade7-public-{subject_name}-{year}-{id}"


def merge_file_list(file_list, type="json"):
    if type == "json":
        data = []
        for file in file_list:
            with open(file, "r", encoding="utf-8") as json_file:
                data.append(json.load(json_file))

        with open(f"grade7-public.json", "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    elif type == "csv":
        df_list = []
        for file in file_list:
            df_list.append(pd.read_csv(file))

        df = pd.concat(df_list, ignore_index=True)
        df.to_csv("grade7-public.csv", index=False)

    else:
        print(f"{type} 형식은 지원하지 않습니다.")


if __name__ == "__main__":
    question_file = None
    answer_file = None
    type = "json"

    extract_data = ExtractGrade7Exam(question_file, answer_file)
    extract_data.run(type)

    csv_file = []
    json_file = []
    merge_file_list(csv_file, "csv")
    merge_file_list(json_file, "json")
