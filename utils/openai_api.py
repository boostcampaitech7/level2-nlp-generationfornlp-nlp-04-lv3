import json
import time
from datetime import datetime
import pytz
from pydantic import BaseModel
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from message_builder import MessageBuilder


MODEL_COSTS = {"gpt-4o-mini": [0.150 / 1000000, 0.600 / 1000000]}


class IsExist(BaseModel):
    is_exist: int


class OpenAIApi:

    def __init__(self, api_key, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def test(self, message):
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=message,
            temperature=0,
            response_format=IsExist,
        )
        return response

    def create_batch_file(
        self,
        message_list,
        id_list=None,
        file_name="batch",
        structured_output_class=None,
    ):

        if id_list is None:
            id_list = [i for i in range(len(message_list))]

        request_list = []
        for id, message in zip(id_list, message_list):
            # API 요청 작성
            request = {
                "custom_id": f"{id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": f"{self.model_name}",
                    "messages": message,
                    "temperature": 0,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stop": None,
                    #'logprobs': True,
                    #'top_logprobs': 10,
                    "n": 1,
                },
            }
            # Structured Output 사용
            if structured_output_class:
                schema = to_strict_json_schema(structured_output_class)
                schema_name = structured_output_class.__name__
                schema["type"] = "object"
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "strict": True,
                        "name": f"{schema_name}",
                        "schema": schema,
                    },
                }
                request["body"]["response_format"] = response_format

            request_list.append(request)

        with open(f"{file_name}.jsonl", "w") as file:
            for request in request_list:
                file.write(json.dumps(request, ensure_ascii=False) + "\n")

    def call_batch(self, input_file):

        output_file = f"{input_file.split('.')[0]}_output.jsonl"
        # 1. 배치 입력 파일 업로드
        batch_input_file = self.client.files.create(
            file=open(input_file, "rb"), purpose="batch"
        )

        # 2. 배치 작업 생성
        batch_job = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        # 3. 배치 작업 상태 확인
        while True:
            batch_status = self.client.batches.retrieve(batch_job.id).status
            # 3.1. 작업 완료 시 결과 파일에 저장
            if batch_status == "completed":
                output_file_id = self.client.batches.retrieve(
                    batch_job.id
                ).output_file_id

                result = self.client.files.content(output_file_id).content.decode(
                    "utf-8"
                )
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(result)
                print("배치 작업이 성공적으로 완료되었습니다.")
                self.calulcate_cost(output_file)
                break

            #           # 3.2. 작업 실패 혹은 오류 발생 시 경고문 출력 후 종료
            elif batch_status in ["failed", "expired", "canceling", "cancelled"]:
                print("배치 작업에서 오류가 발생했습니다!!!")
                break

            # 3.3. 작업 진행 중이면 10초 후 재확인
            else:
                print(f"현재 배치 작업 상태: {batch_status}")
                seoul_time = datetime.now(pytz.timezone("Asia/Seoul"))
                print("현재 시각:", seoul_time.strftime("%Y-%m-%d %H:%M:%S"))
                time.sleep(10)  # 10초 지연 후 상태 재확인

    def calulcate_cost(self, output_file):
        total_cost = 0
        with open(output_file, "r") as file:
            for line in file:
                row = json.loads(line.strip())
                input_cost = (
                    MODEL_COSTS[self.model_name][0]
                    * row["response"]["body"]["usage"]["prompt_tokens"]
                )
                output_cost = (
                    MODEL_COSTS[self.model_name][1]
                    * row["response"]["body"]["usage"]["completion_tokens"]
                )
                total_cost += input_cost + output_cost
        print(f"API 비용: ${total_cost:.5f}")


if __name__ == "__main__":
    message_builder = MessageBuilder()
    # "배경지식 없이 오직 지문을 독해해서 질문을 풀 수 있다면 1 없다면 0을 출력해주세요."
    #
    # "아래 주어진 지문 안에서 문제에 대한 정답이 있다면 1 없다면 0을 출력해주세요."
    id_list, message_list = message_builder.create_message_list(
        system_message="아래 질문의 정답이 지문 안에 있는지에 대한 여부 알려주세요. 있으면 1 없으면 0을 출력해주세요.",
        file_path="/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/data/train.csv",
    )

    openai_api = OpenAIApi(
        api_key="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"
    )
    openai_api.create_batch_file(
        id_list=id_list[:4],
        message_list=message_list[:4],
        structured_output_class=IsExist,
    )
    openai_api.call_batch(
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/batch.jsonl"
    )
    """
    idx = id_list.index("generation-for-nlp-2024")
    response = openai_api.test(message_list[1])
    print(message_list[1])
    print(response.choices[0].message.parsed.is_exist)
    """
