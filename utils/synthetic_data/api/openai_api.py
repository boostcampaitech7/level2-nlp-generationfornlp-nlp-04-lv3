import os
import json
from tqdm import tqdm
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

from api.base import BaseApi


class OpenAIApi(BaseApi):

    def __init__(self, api_key):
        super().__init__()
        self.client = OpenAI(api_key=api_key)

    def test(self, prompt, model_name, structured_output=None):

        response = self.client.beta.chat.completions.parse(
            model=model_name,
            messages=prompt,
            temperature=0,
            **({"response_format": structured_output} if structured_output else {}),
        )
        try:
            response_data = json.loads(
                response.choices[0].to_dict()["message"]["content"]
            )
        except:
            response_data = response.choices[0].to_dict()["message"]["content"]
        return response_data

    def create_batch_file(
        self,
        message_list,
        model_name,
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
                    "model": f"{model_name}",
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

        with open(batch_file, "w") as file:
            for request in request_list:
                file.write(json.dumps(request, ensure_ascii=False) + "\n")

    def call(self, batch_file, batch_size=100):

        # 1. 배치 파일 열기
        with open(batch_file, "r", encoding="utf-8") as file:
            request_list = [json.loads(line) for line in file]

        # 2. API 호출
        batch_idx, response_list = 0, []
        for idx, request in tqdm(
            enumerate(request_list), desc="running...", total=len(request_list)
        ):
            response = self.test(
                request["body"]["messages"],
                request["body"]["model"],
                request["body"].get("response_format", None),
            )

            response_list.append({"id": request["custom_id"], "response": response})
            # 3. 배치 크기만큼 저장되면 파일로 저장
            if batch_size == len(response_list) or idx + 1 == len(request_list):
                sub_batch_file = (
                    f"{os.path.splitext(batch_file)[0]}_output{batch_idx}.jsonl"
                )
                with open(sub_batch_file, "w", encoding="utf-8") as file:
                    for response in response_list:
                        file.write(json.dumps(response, ensure_ascii=False) + "\n")
                response_list = []
                batch_idx += 1

    def call_batch(self, batch_file, batch_size=100):

        # 1. 배치 크기로 데이터 나누고 파일로 저장
        data = []
        with open(batch_file, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line.strip()))

        batch_idx = 1
        sub_batch_file_list = []
        for idx in range(0, len(data), batch_size):
            sub_batch_file = f"{os.path.splitext(batch_file)[0]}_{batch_idx}.jsonl"
            with open(sub_batch_file, "w", encoding="utf-8") as file:
                for item in data[idx : idx + batch_size]:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
            sub_batch_file_list.append(sub_batch_file)
            batch_idx += 1

        # 2. 배치 크기로 나눈 파일로 API 호출
        batch_id_list = []
        for sub_batch_file in sub_batch_file_list:

            # 2.1. 배치 입력 파일 업로드
            batch_input_file = self.client.files.create(
                file=open(sub_batch_file, "rb"), purpose="batch"
            )

            # 2.2. 배치 작업 생성
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            batch_id_list.append(batch_job.id)

        for batch_id in batch_id_list:
            print(f"batch id: {batch_id}")
        return batch_id_list

    def retrieve_batch(self, output_file, batch_id):
        batch_status = self.client.batches.retrieve(batch_id).status

        if batch_status != "completed":
            return False

        # API 결과 파일로 저장
        output_file_id = self.client.batches.retrieve(batch_id).output_file_id
        response_list = self.client.files.content(output_file_id).content.decode(
            "utf-8"
        )
        with open(output_file, "w", encoding="utf-8") as file:
            for response in response_list.splitlines():
                response = json.loads(response)
                try:
                    response_data = json.loads(
                        response["response"]["body"]["choices"][0]["message"]["content"]
                    )
                except:
                    response_data = response["response"]["body"]["choices"][0][
                        "message"
                    ]["content"]
                result = {
                    "id": response["custom_id"],
                    "response": (
                        response_data
                        if response["response"]["status_code"] == 200
                        else None
                    ),
                }
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
        return True
