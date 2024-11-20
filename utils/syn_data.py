import os
from dotenv import load_dotenv

import api
from prompt_builder import SynDataGenPromptBuilder


class SynDataGenerator:

    def __init__(
        self,
        api_key,
        api_type,
        data_file,
        aug_instruction=None,
        cot_instruction=None,
        compare_instruction=None,
    ):

        if hasattr(api, api_type):
            self.api = getattr(api, api_type)(api_key)
        self.prompt_builder = SynDataGenPromptBuilder()
        self.data_file = data_file
        self.api_type = api_type

        self.instructions = {
            "aug": aug_instruction,
            "cot": cot_instruction,
            "compare": compare_instruction,
        }

    def test(self, instruction, idx=0):
        id_list, prompt_list = self.prompt_builder.create_prompt_list(
            instruction, self.data_file
        )
        response = self.api.test(prompt_list[idx])
        return response

    def test_aug(self):
        return self.test(self.instructions["aug"])

    def test_cot(self):
        return self.test(self.instructions["cot"])

    def test_compare(self):
        return self.test(self.instructions["compare"])

    def run(self, instruction, batch_file, type="batch"):
        id_list, prompt_list = self.prompt_builder.create_prompt_list(
            instruction, self.data_file
        )
        batch_file = self.api.create_batch_file(
            id_list=id_list[:2], message_list=prompt_list[:2], batch_file=batch_file
        )
        if type == "batch":
            self.api.call_batch(batch_file)
        else:
            self.api.call(batch_file)

    def augmentation(self, type="batch"):
        self.run(
            self.instructions["aug"],
            f"{data_file.split('/')[-1].split('.')[0]}_{self.api_type}_aug_{type}",
            type,
        )

    def cot(self, type="batch"):
        self.run(
            self.instructions["cot"],
            f"{data_file.split('/')[-1].split('.')[0]}_{self.api_type}_cot_{type}",
            type,
        )

    def compare(self, type="batch"):
        self.run(
            self.instructions["compare"],
            f"{data_file.split('/')[-1].split('.')[0]}_{self.api_type}_compare_{type}",
            type,
        )


if __name__ == "__main__":
    load_dotenv()
    api_type_key_pair = {
        "OpenAIApi": os.getenv("OPENAI_KEY"),
        "GeminiApi": os.getenv("GEMINI_KEY"),
        "ClaudeApi": os.getenv("CLAUDE_KEY"),
    }
    data_file = "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/data/uniform/train.csv"

    aug_prompt = """
    다음 문제와 지문을 참고하여, 문제에 사용된 핵심 개념이나 내용을 바탕으로 새로운 수능형 문제를 작성하시오. 작성할 문제는 다음 조건을 충족해야 합니다:
    1. 문제의 지문과 질문은 기존 문제의 핵심 개념과 관련되어야 합니다.
    2. 선택지는 4개로 구성하며, 각 선택지는 구체적이고 유사하지만, 하나만 정답이 되도록 설계합니다.
    3. 지문과 질문은 수능형 스타일에 맞게 간결하고 논리적으로 작성합니다.

    출력 형태:
    - 새로운 지문
    - 새로운 질문
    - 새로운 선택지 (정답 포함)"""
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

    api_type = ["OpenAIApi", "GeminiApi", "ClaudeApi"]
    i = 1
    syn_data_gen = SynDataGenerator(
        api_type=api_type[i],
        api_key=api_type_key_pair[api_type[i]],
        data_file=data_file,
        aug_instruction=aug_prompt,
        cot_instruction=solving_prompt,
    )
    # print(syn_data_gen.test_cot())
    syn_data_gen.cot(type="not batch")
