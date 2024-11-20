import os
from itertools import islice
from dotenv import load_dotenv
import json
from pydantic import BaseModel
from anthropic import Anthropic
from message_builder import MessageBuilder
from concurrent.futures import ThreadPoolExecutor


MODEL_COSTS = {"claude-3-5-sonnet-20241022": [3.00 / 1000000, 15.00 / 1000000]}


class IsExist(BaseModel):
    is_exist: int


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk


class ClaudeApi:
    def __init__(
        self, api_key, system_message, model_name="claude-3-5-sonnet-20241022"
    ):
        self.system_message = system_message
        self.model_name = model_name
        self.client = Anthropic(api_key=api_key)

    def test(self, message):
        response = self.client.messages.create(
            model=self.model_name,
            system=self.system_message,
            messages=message,
            max_tokens=1024,
            temperature=0,
        )
        return response

    def process_message(self, message, id):
        try:
            response = self.client.messages.create(
                model=self.model_name,
                system=self.system_message,
                messages=message,
                max_tokens=1024,
                temperature=0,
            )

            content_blocks = response.content
            content_text = "\n".join(
                block.text for block in content_blocks if hasattr(block, "text")
            )

            clean_content = content_text.lstrip("0\n\n").strip()

            result = {
                "id": id,
                "content": clean_content,
                "usage": {
                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                },
            }
            return result
        except Exception as e:
            return {"id": id, "error": str(e)}

    def call_batch(
        self, id_list, message_list, output_file="batch_output.jsonl", batch_size=4
    ):
        results = []
        for batch_ids, batch_messages in zip(
            chunked_iterable(id_list, batch_size),
            chunked_iterable(message_list, batch_size),
        ):
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.process_message, message, id)
                    for id, message in zip(batch_ids, batch_messages)
                ]
                for future in futures:
                    results.append(future.result())

            # 결과를 파일에 추가 저장
            with open(output_file, "a", encoding="utf-8") as file:
                for result in results:
                    file.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(
                f"{len(batch_ids)}개의 데이터를 처리하여 결과가 {output_file}에 저장되었습니다."
            )

        self.calculate_cost(output_file)

    def calculate_cost(self, output_file):
        total_cost = 0
        with open(output_file, "r") as file:
            for line in file:
                row = json.loads(line.strip())
                if "usage" in row:
                    # 기본값 처리
                    input_tokens = row["usage"].get("input_tokens", 0)
                    completion_tokens = row["usage"].get("output_tokens", 0)

                    input_cost = MODEL_COSTS[self.model_name][0] * input_tokens
                    output_cost = MODEL_COSTS[self.model_name][1] * completion_tokens

                    total_cost += input_cost + output_cost
        print(f"API 비용: ${total_cost:.5f}")


if __name__ == "__main__":
    # 1. env 파일 불러오기
    load_dotenv()

    # 2. MessageBuilder 인스턴스 생성 - 인자로 claude 설정
    message_builder = MessageBuilder("claude")

    # 3. csv파일을 Claude API Message 형식에 맞게 변환
    id_list, message_list = message_builder.create_message_list(
        file_path="/data/ephemeral/home/ms/level2-nlp-generationfornlp-nlp-04-lv3/data/default/train.csv",
    )

    # 4. system message 설정
    system_message = "아래 질문의 정답이 지문 안에 있는지에 대한 여부를 알려주세요. 있으면 1 없으면 0을 출력해주세요."

    # 5. ClaudeApi 인스턴스 생성 - 인자로 api_key, system_message 설정
    claude_api = ClaudeApi(
        api_key=os.getenv("CLAUDE_API_KEY"), system_message=system_message
    )

    # 6. 배치 방식으로 api 호출
    claude_api.call_batch(
        id_list=id_list,
        message_list=message_list,
        output_file="/data/ephemeral/home/ms/level2-nlp-generationfornlp-nlp-04-lv3/batch_output.jsonl",
        batch_size=4,
    )
