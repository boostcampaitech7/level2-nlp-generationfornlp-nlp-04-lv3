import pdfplumber
from collections import deque
import re
import heapq
from pdfplumber.page import Page
import pandas as pd
import os
from dotenv import load_dotenv


class Extractor:
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
        left_bbox = (0, 75, page_width / 2, page_height)  # 왼쪽 단
        right_bbox = (page_width / 2, 75, page_width, page_height)  # 오른쪽 단

        # 왼쪽 단
        self.texts = deque(page.within_bbox(left_bbox).extract_text_lines())  # 텍스트 줄 단위로 가져오기
        self.images = deque(page.within_bbox(left_bbox).images)
        self.tables = deque(page.within_bbox(left_bbox).find_tables())
        # heapq.heapify(left_images)
        # for image in page.within_bbox(left_bbox).images:  # 이미지 좌표 정보 가져오기
        #     heapq.heappush(left_images, (image.top, image))
        self.find_questions()


        self.texts = deque(page.within_bbox(right_bbox).extract_text_lines())
        self.images = deque(page.within_bbox(right_bbox).images)
        self.tables = deque(page.within_bbox(right_bbox).find_tables())
        self.find_questions()


    def find_questions(self):
        current_questionset = {
            "question": [],
            "paragraph": [],
            "choices": [],
            "question_plus": [],
        }
        key = "question"
        while self.texts:
            if current_questionset["question"] and bool(re.match(r'^\d+\.', self.texts[0]["text"])):
                # 문제셋에 이미지가 포함되지 않으면 데이터프레임에 추가
                if not self.is_with_image(current_questionset) and not self.is_with_table(current_questionset):
                    self.append_questionset(current_questionset)
                # 문제셋 비우기
                for k in current_questionset.keys():
                    current_questionset[k].clear()

            text = self.texts.popleft()
            if bool(re.match(r'^\d+\.', text["text"])):
                key = "question"
            elif text["text"] == "<보기>" or text["text"] == "<보기 1>":
                key = "paragraph"
            elif text["text"] == "<보기 2>":
                key = "question_plus"
            elif bool(re.match(r'^①', text["text"])):
                key = "choices"
            
            current_questionset[key].append(text)
        
        # 마지막 문제셋 처리
        if not self.is_with_image(current_questionset) and not self.is_with_table(current_questionset):
            self.append_questionset(current_questionset)

    
    def is_with_image(self, questionset):
        questionset_top = questionset["question"][0]["top"]
        questionset_bottom = questionset["choices"][-1]["bottom"]
        is_with = False
        while self.images and questionset_top < self.images[0]["top"] < questionset_bottom:
            is_with = True
            self.images.popleft()
        return is_with
    

    def is_with_table(self, questionset):
        questionset_top = questionset["question"][0]["top"]
        questionset_bottom = questionset["choices"][-1]["bottom"]
        is_with = False
        while self.tables and questionset_top < self.tables[0].cells[0][1] < questionset_bottom:
            is_with = True
            self.tables.popleft()
        return is_with
    

    def append_questionset(self, questionset):
        # question
        question = " ".join([text["text"] for text in questionset["question"]])
        question = question.replace("<보기 2>는", "지문은")
        question = question.replace("<보기 2>", "<보기>")
        question = question.replace("<보기 1>", "지문")
        question = question.replace("<보기>는", "지문은")
        question = question.replace("<보기>", "지문")
        self.question_list.append(question)
        # paragraph
        paragraph = " ".join([text["text"] for text in questionset["paragraph"][1:]]) if questionset["paragraph"] else ""
        self.paragraph_list.append(paragraph)
        # choices
        choices = []
        choice = questionset["choices"][0]["text"][2:]
        for text in questionset["choices"][1:]:
            if bool(re.match(r'^[①②③④⑤]', text["text"])):
                if "②" in choice:
                    choice = choice.split("②")
                    for c in choice:
                        choices.append(c.strip())
                else:
                    choices.append(choice)
                choice = text["text"][2:]
            else:
                choice += f" {text['text']}"
        if "④" in choice:
            choice = choice.split("④")
            for c in choice:
                choices.append(c.strip())
        else:
            choices.append(choice)
        self.choices_list.append(choices)
        qeustion_plus = " ".join([text["text"] for text in questionset["question_plus"][1:]]) if questionset["question_plus"] else ""
        self.question_plus_list.append(qeustion_plus)

    
    def save_questions(self):
        ids = []
        problems = []
        answer_list = self.get_answers()

        for i in range(len(self.question_list)):
            question = self.question_list[i]
            question_num = question.split(".")[0]
            ids.append(f"{self.pdf_name}-{question_num}")
            problem = "{'question': '" + question[len(question_num)+1:].strip() + "', 'choices': " + str(self.choices_list[i]) + ", 'answer': " + str(answer_list[int(question_num)-1]) + "}"
            problems.append(problem)

        dataset = pd.DataFrame({
            "id": ids,
            "paragraph": self.paragraph_list,
            "problems": problems,
            "question_plus": self.question_plus_list,
        })

        output_path = os.path.join(os.getenv("ROOT_DIR"), f"aug_data/{pdf_name}.csv")
        dataset.to_csv(output_path, index=False, encoding="utf-8")


    def get_answers(self):
        answer_list = []
        answer_pdf_path = os.path.join(os.getenv("ROOT_DIR"), f"aug_data/{pdf_name}_answer.pdf")
        with pdfplumber.open(answer_pdf_path) as pdf:
            for page in pdf.pages:
                # 페이지 텍스트 추출
                text = page.extract_text()
                if text:
                    # "사회_9급" 포함된 행 찾기
                    for line in text.split("\n"):
                        if "사회_9급 B" in line:
                            # 숫자만 추출
                            parts = line.split()
                            numbers = [part for part in parts if part.isdigit()]
                            answer_list.extend(numbers)
        return answer_list


if __name__ == "__main__":
    load_dotenv()
    pdf_name = "seoul9_social_2020"
    extractor = Extractor(pdf_name)
    extractor.extract_questions()
