import pdfplumber
from collections import deque
import re
from tqdm import tqdm
from pdfplumber.page import Page
import pandas as pd
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod


class Extractor(ABC):
    def __init__(self, pdf_name):
        self.pdf_name = pdf_name
        self.question_list = []
        self.paragraph_list = []
        self.choices_list = []
        self.question_plus_list = []

        self.texts = None
        self.images = None
        self.tables = None

    def extract_questions(self):
        pdf_path = os.path.join(os.getenv("ROOT_DIR"), f"aug_data/{self.pdf_name}.pdf")
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                self.extract_questions_per_page(page)

        self.save_questions()

    def extract_questions_per_page(self, page: Page):
        page_width = page.width
        page_height = page.height

        # 좌/우 단의 좌표 설정
        left_bbox = (0, 0, page_width / 2, page_height)  # 왼쪽 단
        right_bbox = (page_width / 2, 0, page_width, page_height)  # 오른쪽 단

        # 왼쪽 단
        self.texts = deque([])
        for text in page.within_bbox(left_bbox).extract_text_lines():
            self.texts.append([0, text])  # left, text
        self.images = deque([])
        for image in page.within_bbox(left_bbox).images:
            self.images.append([0, image])
        self.tables = deque([])
        for table in page.within_bbox(left_bbox).find_tables():
            self.tables.append([0, table])

        for text in page.within_bbox(right_bbox).extract_text_lines():
            self.texts.append([1, text])  # right, text
        for image in page.within_bbox(right_bbox).images:
            self.images.append([1, image])
        for table in page.within_bbox(right_bbox).find_tables():
            self.tables.append([1, table])

        self.find_questions()

    @abstractmethod
    def find_questions(self):
        pass

    def is_with_image(self, top_text, bottom_text):
        is_with = False
        top = top_text[1]["top"]
        top_position = top_text[0]
        bottom = bottom_text[1]["bottom"]
        bottom_position = bottom_text[0]

        # 영역이 왼쪽 단에서 오른쪽 단까지 이어져 있는 경우
        if top_position != bottom_position:
            while self.images:
                if top_position > self.images[0][0] or top > self.images[0][1]["top"]:
                    self.images.popleft()
                elif (
                    top_position == self.images[0][0]
                    and top < self.images[0][1]["top"] < 10000
                ):
                    is_with = True
                    self.images.popleft()
                else:
                    break
            while self.images:
                if bottom_position > self.images[0][0] or 0 > self.images[0][1]["top"]:
                    self.images.popleft()
                elif (
                    bottom_position == self.images[0][0]
                    and 0 < self.images[0][1]["top"] < bottom
                ):
                    is_with = True
                    self.images.popleft()
                else:
                    break
        else:
            while self.images:
                if top_position > self.images[0][0] or top > self.images[0][1]["top"]:
                    self.images.popleft()
                elif (
                    top_position == self.images[0][0]
                    and top < self.images[0][1]["top"] < bottom
                ):
                    is_with = True
                    self.images.popleft()
                else:
                    break
        return is_with

    def is_with_table(self, top_text, bottom_text):
        is_with = False
        top = top_text[1]["top"]
        top_position = top_text[0]
        bottom = bottom_text[1]["bottom"]
        bottom_position = bottom_text[0]

        # 영역이 왼쪽 단에서 오른쪽 단까지 이어져 있는 경우
        if top_position != bottom_position:
            while self.tables:
                if (
                    top_position > self.tables[0][0]
                    or top > self.tables[0][1].cells[0][1]
                ):
                    self.tables.popleft()
                elif (
                    top_position == self.tables[0][0]
                    and top < self.tables[0][1].cells[0][1] < 10000
                ):
                    is_with = True
                    self.tables.popleft()
                else:
                    break
            while self.tables:
                if (
                    bottom_position > self.tables[0][0]
                    or 0 > self.tables[0][1].cells[0][1]
                ):
                    self.tables.popleft()
                elif (
                    bottom_position == self.tables[0][0]
                    and 0 < self.tables[0][1].cells[0][1] < bottom
                ):
                    is_with = True
                    self.tables.popleft()
                else:
                    break
        else:
            while self.tables:
                if (
                    top_position > self.tables[0][0]
                    or top > self.tables[0][1].cells[0][1]
                ):
                    self.tables.popleft()
                elif (
                    top_position == self.tables[0][0]
                    and top < self.tables[0][1].cells[0][1] < bottom
                ):
                    is_with = True
                    self.tables.popleft()
                else:
                    break
        return is_with

    def append_questionset(self, questionset):
        # question
        question = " ".join([text[1]["text"] for text in questionset["question"]])
        self.question_list.append(question)
        # paragraph
        paragraph = (
            " ".join([text[1]["text"] for text in questionset["paragraph"]])
            if questionset["paragraph"]
            else ""
        )
        self.paragraph_list.append(paragraph)
        # choices
        choices = []
        choice = questionset["choices"][0][1]["text"][1:]
        for text in questionset["choices"][1:]:
            if bool(re.match(r"^[①②③④⑤]", text[1]["text"])):
                if re.search(r"[①②③④⑤]", choice):
                    choice = re.split(r"[②③④⑤]", choice)
                    for c in choice:
                        choices.append(c.strip())
                else:
                    choices.append(choice)
                choice = text[1]["text"][1:]
            else:
                choice += f" {text[1]['text']}"
        if re.search(r"[①②③④⑤]", choice):
            choice = re.split(r"[②③④⑤]", choice)
            for c in choice:
                choices.append(c.strip())
        else:
            choices.append(choice)
        self.choices_list.append(choices)
        qeustion_plus = (
            " ".join([text[1]["text"] for text in questionset["question_plus"]])
            if questionset["question_plus"]
            else ""
        )
        self.question_plus_list.append(qeustion_plus)

    def save_questions(self):
        ids = []
        problems = []
        answer_list = self.get_answers()

        id_prefix = ""
        pdf_name_split = self.pdf_name.split("_")
        exam_name, year = pdf_name_split[0], pdf_name_split[-1]
        if exam_name == "seoul9":
            id_prefix = f"{exam_name}-social-{year}"
        elif exam_name == "PSAT":
            id_prefix = f"{exam_name}-korean-{year}"
        elif exam_name == "police":
            id_prefix = f"{exam_name}-korean-{year}"

        for i in range(len(self.question_list)):
            question = self.question_list[i]
            question_num = question.split(".")[0]
            question_start = len(question_num)
            if "문" in question_num:
                question_num = question_num.split(" ")[1]
            ids.append(f"{id_prefix}-{question_num}")
            problem = (
                "{'question': '"
                + question[question_start + 1 :].strip()
                + "', 'choices': "
                + str(self.choices_list[i])
                + ", 'answer': "
                + str(answer_list[int(question_num) - 1])
                + "}"
            )
            problems.append(problem)

        dataset = pd.DataFrame(
            {
                "id": ids,
                "paragraph": self.paragraph_list,
                "problems": problems,
                "question_plus": self.question_plus_list,
            }
        )

        output_path = os.path.join(os.getenv("ROOT_DIR"), f"aug_data/{pdf_name}.csv")
        dataset.to_csv(output_path, index=False, encoding="utf-8")
        print(f"{pdf_name}.csv가 저장되었습니다.")

    @abstractmethod
    def get_answers(self):
        pass


class Seoul9Extractor(Extractor):
    def __init__(self, pdf_name):
        super().__init__(pdf_name)

    def find_questions(self):
        current_questionset = {
            "question": [],
            "paragraph": [],
            "choices": [],
            "question_plus": [],
        }
        key = "question"
        while self.texts:
            if (
                current_questionset["question"]
                and bool(re.match(r"^\d+\.", self.texts[0][1]["text"]))
                and key == "choices"
            ):
                # 문제셋에 이미지가 포함되지 않으면 데이터프레임에 추가
                if not self.is_with_image(
                    current_questionset["question"][0],
                    current_questionset["choices"][-1],
                ) and not self.is_with_table(
                    current_questionset["question"][0],
                    current_questionset["choices"][-1],
                ):
                    self.append_questionset(current_questionset)
                # 문제셋 비우기
                for k in current_questionset.keys():
                    current_questionset[k].clear()

            text = self.texts.popleft()
            if "9급" in text[1]["text"]:
                continue
            if bool(re.match(r"^\d+\.", text[1]["text"])) and key == "choices":
                key = "question"
            elif text[1]["text"] == "<보기>" or text[1]["text"] == "<보기 1>":
                key = "paragraph"
            elif text[1]["text"] == "<보기 2>":
                key = "question_plus"
            elif bool(re.match(r"^①", text[1]["text"])):
                key = "choices"

            current_questionset[key].append(text)

        # 마지막 문제셋 처리
        if not self.is_with_image(
            current_questionset["question"][0], current_questionset["choices"][-1]
        ) and not self.is_with_table(
            current_questionset["question"][0], current_questionset["choices"][-1]
        ):
            self.append_questionset(current_questionset)

    def append_questionset(self, questionset):
        question = " ".join([text[1]["text"] for text in questionset["question"]])
        question = question.replace("<보기 2>는", "지문은")
        question = question.replace("<보기 2>", "<보기>")
        question = question.replace("<보기 1>", "지문")
        question = question.replace("<보기>는", "지문은")
        question = question.replace("<보기>", "지문")
        self.question_list.append(question)
        # paragraph
        paragraph = (
            " ".join([text[1]["text"] for text in questionset["paragraph"]])
            if questionset["paragraph"]
            else ""
        )
        self.paragraph_list.append(paragraph)
        # choices
        choices = []
        choice = questionset["choices"][0][1]["text"][1:]
        for text in questionset["choices"][1:]:
            if bool(re.match(r"^[①②③④⑤]", text[1]["text"])):
                if re.search(r"[①②③④⑤]", choice):
                    choice = re.split(r"[②③④⑤]", choice)
                    for c in choice:
                        choices.append(c.strip())
                else:
                    choices.append(choice)
                choice = text[1]["text"][1:]
            else:
                choice += f" {text[1]['text']}"
        if re.search(r"[①②③④⑤]", choice):
            choice = re.split(r"[②③④⑤]", choice)
            for c in choice:
                choices.append(c.strip())
        else:
            choices.append(choice)
        self.choices_list.append(choices)
        qeustion_plus = (
            " ".join([text[1]["text"] for text in questionset["question_plus"]])
            if questionset["question_plus"]
            else ""
        )
        self.question_plus_list.append(qeustion_plus)

    def get_answers(self):
        answer_list = []
        answer_pdf_path = os.path.join(
            os.getenv("ROOT_DIR"), f"aug_data/{pdf_name}_answer.pdf"
        )
        answer_key_dict = {
            "seoul9_social_2019": "사회_9급 A",
            "seoul9_social_2020": "사회_9급 B",
            "seoul9_social_2021": "사회(9급) A",
        }

        with pdfplumber.open(answer_pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for line in text.split("\n"):
                        if answer_key_dict[self.pdf_name] in line:
                            parts = line.split()
                            numbers = [part for part in parts if part.isdigit()]
                            answer_list.extend(numbers)
        return answer_list


class PSATExtractor(Extractor):
    def __init__(self, pdf_name):
        super().__init__(pdf_name)

    def find_questions(self):
        current_questionset = {
            "question": [],
            "paragraph": [],
            "choices": [],
            "question_plus": [],
        }
        # 1지문 2문제
        if "다음 글을 읽고 물음에 답하시오" in self.texts[1][1]["text"]:
            key = "paragraph"
            exclude_all = False
            while self.texts:
                if (
                    current_questionset["paragraph"]
                    and bool(
                        re.match(r"^(문 \d+\.)|(^\d+\. )", self.texts[0][1]["text"])
                    )
                    and key == "paragraph"
                ):
                    # 지문 안에 이미지, 표가 있으면 두 문제 다 제외
                    if self.is_with_image(
                        current_questionset["paragraph"][0],
                        current_questionset["paragraph"][-1],
                    ) or self.is_with_table(
                        current_questionset["paragraph"][0],
                        current_questionset["paragraph"][-1],
                    ):
                        exclude_all = True
                        break
                elif (
                    current_questionset["question"]
                    and bool(
                        re.match(r"^(문 \d+\.)|(^\d+\. )", self.texts[0][1]["text"])
                    )
                    and key == "choices"
                ):
                    # 문제 안에 이미지, 표가 없으면 추가
                    if not self.is_with_image(
                        current_questionset["question"][0],
                        current_questionset["choices"][-1],
                    ) and not self.is_with_table(
                        current_questionset["question"][0],
                        current_questionset["choices"][-1],
                    ):
                        self.append_questionset(current_questionset)
                    # 지문은 남기고 문제셋 비우기
                    current_questionset["question"].clear()
                    current_questionset["question_plus"].clear()
                    current_questionset["choices"].clear()

                text = self.texts.popleft()
                if (
                    "언어논리영역" in text[1]["text"]
                    or "국가공무원 5급" in text[1]["text"]
                    or "다음 글을 읽고 물음에 답하시오" in text[1]["text"]
                ):
                    continue
                if bool(re.match(r"^(문 \d+\.)|(^\d+\. )", text[1]["text"])) and (
                    key == "choices" or key == "paragraph"
                ):
                    key = "question"
                elif text[1]["text"] == "<보기>" or text[1]["text"] == "<보 기>":
                    key = "question_plus"
                elif bool(re.match(r"^①", text[1]["text"])):
                    key = "choices"

                if not (text[1]["text"] == "<보 기>" or text[1]["text"] == "<보기>"):
                    current_questionset[key].append(text)
                    if key == "question" and bool(re.search(r"\?$", text[1]["text"])):
                        key = "question_plus"

            if not exclude_all:
                # 마지막 문제셋 처리
                if not self.is_with_image(
                    current_questionset["question"][0],
                    current_questionset["choices"][-1],
                ) and not self.is_with_table(
                    current_questionset["question"][0],
                    current_questionset["choices"][-1],
                ):
                    self.append_questionset(current_questionset)
        # 1지문 1문제
        else:
            key = "question"
            while self.texts:
                if (
                    current_questionset["question"]
                    and bool(
                        re.match(r"^(문 \d+\.)|(^\d+\. )", self.texts[0][1]["text"])
                    )
                    and key == "choices"
                ):
                    # 문제셋에 이미지가 포함되지 않으면 데이터프레임에 추가
                    if not self.is_with_image(
                        current_questionset["question"][0],
                        current_questionset["choices"][-1],
                    ) and not self.is_with_table(
                        current_questionset["question"][0],
                        current_questionset["choices"][-1],
                    ):
                        self.append_questionset(current_questionset)
                    # 문제셋 비우기
                    for k in current_questionset.keys():
                        current_questionset[k].clear()

                text = self.texts.popleft()
                if (
                    "언어논리영역" in text[1]["text"]
                    or "국가공무원 5급" in text[1]["text"]
                ):
                    continue
                if (
                    bool(re.match(r"^(문 \d+\.)|(^\d+\. )", text[1]["text"]))
                    and key == "choices"
                ):
                    key = "question"
                elif text[1]["text"] == "<보기>" or text[1]["text"] == "<보 기>":
                    key = "question_plus"
                elif bool(re.match(r"^①", text[1]["text"])):
                    key = "choices"

                if not (text[1]["text"] == "<보 기>" or text[1]["text"] == "<보기>"):
                    current_questionset[key].append(text)
                if key == "question" and bool(re.search(r"\?$", text[1]["text"])):
                    key = "paragraph"

            # 마지막 문제셋 처리
            if not self.is_with_image(
                current_questionset["question"][0], current_questionset["choices"][-1]
            ) and not self.is_with_table(
                current_questionset["question"][0], current_questionset["choices"][-1]
            ):
                self.append_questionset(current_questionset)

    def get_answers(self):
        answer_list = [0] * 40
        answer_pdf_path = os.path.join(
            os.getenv("ROOT_DIR"), f"aug_data/{pdf_name}_answer.pdf"
        )
        answer_key_dict = {
            "PSAT_2020": "영역 언어논리영역 책형 나 책형",
            "PSAT_2021": "영역 언어논리영역 책형 가 책형",
            "PSAT_2022": "영역 언어논리영역 책형 ㉯ 책형",
            "PSAT_2024": "영역 언어논리영역 책형 ㉮ 책형",
            "PSAT_2023": "영역 언어논리영역 책형 ㉯ 책형",
        }

        with pdfplumber.open(answer_pdf_path) as pdf:
            for page in pdf.pages:
                texts = deque(page.extract_text_lines())
                while texts:
                    text = texts.popleft()
                    if answer_key_dict[self.pdf_name] in text["text"]:
                        texts.popleft()
                        while texts:
                            text = texts.popleft()
                            qid1, a1, qid2, a2 = map(int, text["text"].split())
                            answer_list[qid1 - 1] = a1
                            answer_list[qid2 - 1] = a2
        return answer_list


class PoliceExtractor(Extractor):
    def __init__(self, pdf_name):
        super().__init__(pdf_name)

    def extract_questions(self):
        pdf_path = os.path.join(os.getenv("ROOT_DIR"), f"aug_data/{self.pdf_name}.pdf")
        self.texts = deque([])
        self.images = deque([])
        self.tables = deque([])
        # 경찰대 시험지는 지문이 페이지를 넘겨서 이어지는 경우가 존재하므로 한 번에 모든 페이지의 텍스트를 다 받아놓고 시작
        page_start = 0 if "2021" in self.pdf_name else 1
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(page_start, len(pdf.pages)):
                page = pdf.pages[i]
                page_width = page.width
                page_height = page.height

                # 좌/우 단의 좌표 설정
                left_bbox = (0, 0, page_width / 2, page_height - 50)  # 왼쪽 단
                right_bbox = (
                    page_width / 2,
                    0,
                    page_width,
                    page_height - 50,
                )  # 오른쪽 단

                # 왼쪽 단
                for text in page.within_bbox(left_bbox).extract_text_lines():
                    self.texts.append([i * 2, text])  # left, text
                for image in page.within_bbox(left_bbox).images:
                    self.images.append([i * 2, image])
                for table in page.within_bbox(left_bbox).find_tables():
                    self.tables.append([i * 2, table])

                for text in page.within_bbox(right_bbox).extract_text_lines():
                    self.texts.append([i * 2 + 1, text])  # right, text
                for image in page.within_bbox(right_bbox).images:
                    self.images.append([i * 2 + 1, image])
                for table in page.within_bbox(right_bbox).find_tables():
                    self.tables.append([i * 2 + 1, table])

            # 문제지 맨 앞, 맨 뒤 안내문 제거
            self.texts.popleft()
            self.texts.popleft()
            if "하나만 고르시오" in self.texts[0][1]["text"]:
                self.texts.popleft()
            self.texts.pop()
            self.texts.pop()
            self.find_questions()
        self.save_questions()

    def find_questions(self):
        current_questionset = {
            "question": [],
            "paragraph": [],
            "choices": [],
            "question_plus": [],
        }
        key = "question"
        while self.texts:
            # 1지문 N문제
            if "다음 글을 읽고 물음에 답하시오" in self.texts[0][1]["text"]:
                key = "paragraph"
                text = self.texts.popleft()
                question_range = re.split(r"[～∼~]", text[1]["text"].split("]")[0])
                start_question_id = int(question_range[0][1:])
                end_question_id = int(question_range[1])

                # 지문 텍스트 수집
                while self.texts:
                    if not bool(re.match(r"^\d+\.", self.texts[0][1]["text"])):
                        text = self.texts.popleft()
                        current_questionset["paragraph"].append(text)
                    else:
                        break

                # 지문에 이미지 포함되면 딸린 문제 전부 제외하기 위해 flag 설정
                exclude_all = False
                if self.is_with_image(
                    current_questionset["paragraph"][0],
                    current_questionset["paragraph"][-1],
                ) and self.is_with_table(
                    current_questionset["paragraph"][0],
                    current_questionset["paragraph"][-1],
                ):
                    exclude_all = True

                # 지문에 딸린 문제 수만큼 문제 수집
                for _ in range(end_question_id - start_question_id + 1):
                    key = "question"
                    while self.texts:
                        if (
                            current_questionset["question"]
                            and (
                                bool(re.match(r"^\d+\.", self.texts[0][1]["text"]))
                                or "다음 글을 읽고 물음에 답하시오"
                                in self.texts[0][1]["text"]
                            )
                            and key == "choices"
                        ):
                            if (
                                not exclude_all
                                and not self.is_with_image(
                                    current_questionset["question"][0],
                                    current_questionset["choices"][-1],
                                )
                                and not self.is_with_table(
                                    current_questionset["question"][0],
                                    current_questionset["choices"][-1],
                                )
                            ):
                                self.append_questionset(current_questionset)
                            # 지문은 남기고 문제셋 비우기
                            current_questionset["question"].clear()
                            current_questionset["question_plus"].clear()
                            current_questionset["choices"].clear()
                            break

                        text = self.texts.popleft()
                        text[1]["text"] = text[1]["text"].replace("[3점]", "").strip()
                        text[1]["text"] = text[1]["text"].replace("윗글", "지문")
                        if text[1]["text"] == "보 기" or text[1]["text"] == "<보기>":
                            key = "question_plus"
                        elif bool(re.match(r"^①", text[1]["text"])):
                            key = "choices"

                        if (
                            text[1]["text"] != "보 기"
                            and text[1]["text"] != "<보기>"
                            and "시험 (국 어)" not in text[1]["text"]
                        ):
                            current_questionset[key].append(text)
                        if key == "question" and bool(
                            re.search(r"\?$", text[1]["text"])
                        ):
                            key = "question_plus"

                    if (
                        current_questionset["question"]
                        and not exclude_all
                        and not self.is_with_image(
                            current_questionset["question"][0],
                            current_questionset["choices"][-1],
                        )
                        and not self.is_with_table(
                            current_questionset["question"][0],
                            current_questionset["choices"][-1],
                        )
                    ):
                        self.append_questionset(current_questionset)
                # 문제 수집 끝나면 current_questionset 비우기
                for k in current_questionset.keys():
                    current_questionset[k].clear()
                key = "question"

            # 1지문 1문제
            else:
                text = self.texts.popleft()
                text[1]["text"] = text[1]["text"].replace("[3점]", "").strip()
                text[1]["text"] = text[1]["text"].replace("<보기>를", "지문을")
                text[1]["text"] = text[1]["text"].replace("<보기>", "지문")
                if bool(re.match(r"^①", text[1]["text"])):
                    key = "choices"

                if (
                    text[1]["text"] != "보 기"
                    and text[1]["text"] != "<보기>"
                    and "시험 (국 어)" not in text[1]["text"]
                ):
                    current_questionset[key].append(text)
                if key == "question" and bool(re.search(r"\?$", text[1]["text"])):
                    key = "paragraph"

                # 문제 수집 완료
                if (
                    current_questionset["question"]
                    and self.texts
                    and (
                        bool(re.match(r"^\d+\.", self.texts[0][1]["text"]))
                        or "다음 글을 읽고 물음에 답하시오" in self.texts[0][1]["text"]
                    )
                    and key == "choices"
                ):
                    if not self.is_with_image(
                        current_questionset["question"][0],
                        current_questionset["choices"][-1],
                    ) and not self.is_with_table(
                        current_questionset["question"][0],
                        current_questionset["choices"][-1],
                    ):
                        self.append_questionset(current_questionset)

                    for k in current_questionset.keys():
                        current_questionset[k].clear()
                    key = "question"

    def get_answers(self):
        answer_list = []
        answer_pdf_path = os.path.join(
            os.getenv("ROOT_DIR"), f"aug_data/{pdf_name}_answer.pdf"
        )

        with pdfplumber.open(answer_pdf_path) as pdf:
            for page in pdf.pages:
                texts = deque(page.extract_text_lines())
                while texts:
                    text = texts.popleft()
                    if "국 어" in text["text"]:
                        while texts:
                            text = texts.popleft()
                            if "정 답" in text["text"]:
                                nums = text["text"].split()
                                for num in nums:
                                    if num == "①" or num == "1":
                                        answer_list.append(1)
                                    elif num == "②" or num == "2":
                                        answer_list.append(2)
                                    elif num == "③" or num == "3":
                                        answer_list.append(3)
                                    elif num == "④" or num == "4":
                                        answer_list.append(4)
                                    elif num == "⑤" or num == "5":
                                        answer_list.append(5)
        return answer_list


if __name__ == "__main__":
    load_dotenv()
    seoul9_social = [
        "seoul9_social_2019",
        "seoul9_social_2020",
        "seoul9_social_2021",
    ]
    for pdf_name in tqdm(seoul9_social):
        extractor = Seoul9Extractor(pdf_name)
        extractor.extract_questions()

    for i in tqdm(range(0, 1)):
        pdf_name = f"PSAT_202{i}"
        extractor = PSATExtractor(pdf_name)
        extractor.extract_questions()

    for i in tqdm(range(1, 6)):
        pdf_name = f"police_202{i}"
        extractor = PoliceExtractor(pdf_name)
        extractor.extract_questions()
