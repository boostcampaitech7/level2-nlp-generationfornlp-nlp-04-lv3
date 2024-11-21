import json
from tqdm import tqdm
from openai import OpenAI

from api.base import BaseApi, MODEL_COSTS


class GeminiApi(BaseApi):

    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        super().__init__(api_key, model_name)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/",
        )

    def test(self, message):
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=message,
            temperature=0,
        )
        return response

    def create_batch_file(
        self, message_list, batch_file, id_list=None, structured_output=None
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

        with open(f"{batch_file}.jsonl", "w") as file:
            for request in request_list:
                file.write(json.dumps(request, ensure_ascii=False) + "\n")

        return f"{batch_file}.jsonl"

    def call(self, batch_file, batch_size=100):

        # 1. 배치 파일 열기
        with open(batch_file, "r", encoding="utf-8") as file:
            request_list = [json.loads(line) for line in file]

        # 2. API 호출
        batch_idx, response_list = 0, []
        for idx, request in tqdm(
            enumerate(request_list), desc="running...", total=len(request_list)
        ):
            response = self.test(request["message"])
            response_list.append(
                {
                    "id": request["id"],
                    "response": response.to_dict(),
                }
            )

            # 3. 배치 크기만큼 저장되면 파일로 저장
            if batch_size == len(response_list) or idx + 1 == len(request_list):
                sub_batch_file = f"{batch_file.split('.')[0]}_{batch_idx}.jsonl"
                with open(sub_batch_file, "w", encoding="utf-8") as file:
                    for response in response_list:
                        file.write(
                            json.dumps(response, ensure_ascii=False, indent=4) + "\n"
                        )
                response_list = []
                batch_idx += 1

    def call_batch(self, batch_file, batch_size=100):
        raise NotImplementedError
