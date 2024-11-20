import json
import time
import pytz
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

from api.base import BaseApi, MODEL_COSTS


class OpenAIApi(BaseApi):

    def __init__(self, api_key, model_name="gpt-4o-mini"):
        super().__init__(api_key, model_name)
        self.client = OpenAI(api_key=self.api_key)

    def test(self, message):
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=message,
            temperature=0,
        )
        return response

    def create_batch_file(
        self,
        message_list,
        batch_file,
        id_list=None,
        structured_output=None,
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
                    "n": 1,
                },
            }
            # Structured Output 사용
            if structured_output:
                schema = to_strict_json_schema(structured_output)
                schema_name = structured_output.__name__
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

        with open(f"{batch_file}.jsonl", "w") as file:
            for request in request_list:
                file.write(json.dumps(request, ensure_ascii=False) + "\n")

        return f"{batch_file}.jsonl"

    def call(self, batch_file):

        output_file = f"{batch_file.split('.')[0]}_output.jsonl"

        # 1. 배치 파일 열기
        with open(batch_file, "r", encoding="utf-8") as file:
            request_list = [json.loads(line) for line in file]

        # 2. API 호출
        response_list = []
        for request in tqdm(request_list, desc="running...", total=len(request_list)):
            response = self.test(request["body"]["messages"])
            response.id = request["custom_id"]
            response_list.append(response)

        # 3. 파일로 저장
        with open(output_file, "w", encoding="utf-8") as file:
            for response in response_list:
                file.write(
                    json.dumps(response.to_dict(), ensure_ascii=False, indent=4) + "\n"
                )

    def call_batch(self, batch_file):

        output_file = f"{batch_file.split('.')[0]}_output.jsonl"
        # 1. 배치 입력 파일 업로드
        batch_input_file = self.client.files.create(
            file=open(batch_file, "rb"), purpose="batch"
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

                # 3.1.1. 비용 계산
                self.calulcate_cost(output_file)

                # 3.1.2. 유니코드 -> 한글 전환
                data = []
                with open(output_file, "r", encoding="utf-8") as file:
                    for line in file:
                        data.append(json.loads(line.strip()))
                with open(output_file, "w", encoding="utf-8") as file:
                    for item in data:
                        file.write(
                            json.dumps(item, ensure_ascii=False, indent=4) + "\n"
                        )
                break

            # 3.2. 작업 실패 혹은 오류 발생 시 경고문 출력 후 종료
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


def retrieve_batch(output_file, batch_id):
    client = OpenAI()
    output_file_id = client.batches.retrieve(batch_id).output_file_id

    result = client.files.content(output_file_id).content.decode("utf-8")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(result)
    print("배치 작업이 성공적으로 완료되었습니다.")

    # 3.1.2. 유니코드 -> 한글 전환
    data = []
    with open(output_file, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    with open(output_file, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False, indent=4) + "\n")
