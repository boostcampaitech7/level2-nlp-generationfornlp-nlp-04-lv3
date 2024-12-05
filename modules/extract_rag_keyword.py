from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import pandas as pd
import ast
import os
import re


class KeywordExtractor:
    def __init__(
        self,
        model_name="qwen2.5:32b-instruct-q6_K",
        data_path="data/validation_aug_cot.csv",
    ):  # gemma2:27b
        load_dotenv()
        self.model_name = model_name
        self.data_path = data_path
        self.llm = OllamaLLM(model=self.model_name)
        ROOT_DIR = os.getenv("ROOT_DIR")
        self.test = pd.read_csv(os.path.join(ROOT_DIR, self.data_path))

    def generate_prompt(self, question, input_text, choices):
        return f"""
        당신은 교육 콘텐츠를 주어진 분야로 분류하고, Retrieval-Augmented Generation (RAG)을 위한 관련 키워드를 추출하는 지능적이고 정확한 어시스턴트입니다.

        ### 작업 내용
        1. 입력된 데이터가 사회 분야에 속하는지 판단하세요. 사회 분야는 한국 CSAT(수능)의 사회 과목 기준을 따르며, 다음과 같은 과목이 포함됩니다:
        - 정치와 법
        - 경제
        - 심리
        - 한국사
        2. 사회 분야로 분류된다면, 데이터에서 질문과 밀접한 관련이 있는 3개의 구체적인 키워드를 추출하세요.
        3. 사회 분야가 아닌 경우, `is_social` 값을 0으로 설정하고 키워드는 비워둡니다.

        ### 입력 데이터
        - 질문: {question}
        - 선택지: {choices}
        - 지문: {input_text}

        ### 출력 형식
        ```plaintext
        키워드: [키워드1, 키워드2, 키워드3]
        is_social: 0 또는 1
        ```

        응답은 반드시 응답 형식을 지켜야 하며, 그 이외의 정보는 응답에 포함하면 안됩니다. 응답에 주석정보는 넣으면 안됩니다.
        사회 분야가 아니면 반드시 is_social 값을 0으로 하세요.
        """

    def extract_keywords_and_social(self):
        # question
        self.test["question"] = self.test["problems"].apply(
            lambda x: (
                ast.literal_eval(x).get("question") if isinstance(x, str) else None
            )
        )
        # choices
        self.test["choices"] = self.test["problems"].apply(
            lambda x: ast.literal_eval(x).get("choices") if isinstance(x, str) else None
        )

        is_social = []
        keywords = []

        for index, row in self.test.iterrows():
            input_text = row["paragraph"]
            question = row["question"]
            choices = row["choices"]

            formatted_prompt = self.generate_prompt(question, input_text, choices)

            # Ollama 모델에 프롬프트 전달하여 텍스트 복원
            response = self.llm(formatted_prompt)

            # 응답 내용 출력 for 디버깅
            print("응답:", response)

            keyword_match = re.search(
                r"키워드: \[([^\]]+)\]", response
            )  # 키워드를 리스트 형식으로 추출
            is_social_match = re.search(r"is_social: (\d)", response)

            if is_social_match:
                is_social_value = int(is_social_match.group(1))

                if is_social_value == 1 and keyword_match:
                    # 쉼표로 구분된 키워드를 리스트로 변환
                    extracted_keywords = [
                        keyword.strip() for keyword in keyword_match.group(1).split(",")
                    ]
                else:
                    extracted_keywords = []
            else:
                extracted_keywords = []
                is_social_value = None

            # 결과 저장
            keywords.append(extracted_keywords)
            print("키워드:", extracted_keywords)
            is_social.append(is_social_value)
            print("사회인가:", is_social_value)

            # 요청 간에 약간의 대기 시간을 두어 API 호출 과부하를 방지
            # time.sleep(0.2) # 필요시 조정 가능

        return keywords, is_social


# 클래스 사용 예시 - 키워드, 사회여부 추출
extractor = KeywordExtractor(
    model_name="qwen2.5:32b-instruct-q6_K", data_path="data/validation_aug_cot.csv"
)  # gemma2:27b
keywords, is_social = extractor.extract_keywords_and_social()


# CSV 파일로 저장
load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
test = pd.read_csv(os.path.join(ROOT_DIR, "data/validation_aug_cot.csv"))
test["keywords"] = keywords
test["is_social"] = is_social
output_path = os.path.join(ROOT_DIR, f"RAG/qwen_validation_cot_keyword_2.csv")
test.to_csv(output_path, index=False)
print("추출한 키워드를 확인하기 위한 validation 데이터가 CSV 파일로 저장되었습니다.")
