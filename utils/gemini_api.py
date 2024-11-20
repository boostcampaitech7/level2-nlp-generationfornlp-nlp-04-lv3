import json
from tqdm import tqdm
import time
from datetime import datetime
from pydantic import BaseModel
from openai import OpenAI
from prompt_builder import SynDataGenPromptBuilder


MODEL_COSTS = {"gemini-1.5-flash": [0.075 / 1000000, 0.30 / 1000000]}


class GeminiApi:

    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/",
        )

    def test(self, message):
        response = self.client.beta.chat.completions.parse(
            model="gemini-1.5-flash",
            messages=message,
            temperature=0,
        )
        return response

    def create_batch_file(
        self,
        message_list,
        id_list=None,
        file_name="batch",
    ):
        if id_list is None:
            id_list = [i for i in range(len(message_list))]

        request_list = []
        for id, message in zip(id_list, message_list):
            request = {
                "id": id,
                "message": message,
            }
            request_list.append(request)

        with open(f"{file_name}.jsonl", "w") as file:
            for request in request_list:
                file.write(json.dumps(request, ensure_ascii=False) + "\n")

        return f"{file_name}.jsonl"

    def call_batch(self, input_file):

        with open(input_file, "r", encoding="utf-8") as file:
            request_list = [json.loads(line) for line in file]

        response_list = []
        for request in tqdm(request_list, desc="running...", total=len(request_list)):
            response = self.test(request["message"])
            response_list.append(
                {
                    "id": request["id"],
                    "response": response.to_dict(),
                }
            )

        output_file = f"{input_file.split('.')[0]}_output.jsonl"
        with open(output_file, "w", encoding="utf-8") as file:
            for response in response_list:
                file.write(json.dumps(response, ensure_ascii=False, indent=4) + "\n")


if __name__ == "__main__":
    # aug_prompt = "아래 문제에서 사용된 개념으로 새로운 수능형 문제를 만들어주세요. 이때, 선지 중 하나만 정답이 될 수 있도록 해주세요."
    aug_prompt = """
    다음 문제와 지문을 참고하여, 문제에 사용된 핵심 개념이나 내용을 바탕으로 새로운 수능형 문제를 작성하시오. 작성할 문제는 다음 조건을 충족해야 합니다:
    1. 문제의 지문과 질문은 기존 문제의 핵심 개념과 관련되어야 합니다.
    2. 선택지는 4개로 구성하며, 각 선택지는 구체적이고 유사하지만, 하나만 정답이 되도록 설계합니다.
    3. 지문과 질문은 수능형 스타일에 맞게 간결하고 논리적으로 작성합니다.

    출력 형태:
    - 새로운 지문
    - 새로운 질문
    - 새로운 선택지 (정답 포함)"""
    # solving_prompt = "아래 문제를 보고 적절한 문제풀이를 작성해주세요."
    solving_prompt = """프롬프트:
    다음 수능형 문제를 읽고 적절한 문제 풀이를 작성하시오. 풀이 과정은 다음과 같은 구조를 따르도록 하세요:

    지문 분석:
    - 지문의 핵심 내용을 요약하고, 문제 해결에 필요한 중요한 단서를 명확히 파악합니다.
    - 지문 속 키워드나 개념이 어떤 맥락에서 제시되었는지 서술합니다.

    선택지 검토:
    - 각 선택지를 하나씩 검토하며, 옳고 그름을 판단하는 근거를 명확히 제시합니다.
    - 관련된 지문 내용과 역사적/과학적/문학적 배경지식을 활용하여 논리적으로 설명합니다.

    정답 도출:
    -  선택지 중 가장 적합한 답을 선택하고, 이를 지문 내용과 문제 요구 사항에 근거하여 명확히 설명합니다.

    출력 형태:
    - 문제 풀이 과정을 논리적으로 나누어 작성합니다.
    - 각 단계마다 문제 풀이의 이유를 분명히 밝히며, 최종적으로 정답을 명시합니다."""

    prompt_builder = SynDataGenPromptBuilder()
    id_list, message_list = prompt_builder.create_prompt_list(
        instruction=aug_prompt,
        data_file="/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/data/uniform/all.csv",
    )
    print(message_list[0][1]["content"])
    gemini_api = GeminiApi(api_key="AIzaSyATKHUtZ5iqK2Ey11ke-N9mxehsXH1YCqI")
    gemini_api.create_batch_file(message_list[:1])
    gemini_api.call_batch(
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/batch.jsonl"
    )
