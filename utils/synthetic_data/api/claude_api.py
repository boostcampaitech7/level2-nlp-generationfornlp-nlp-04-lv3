import os
import json
from tqdm import tqdm
from anthropic import Anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

from api.base import BaseApi
from api.utils import remove_titles


class ClaudeApi(BaseApi):
    def __init__(self, api_key):
        super().__init__()
        self.client = Anthropic(api_key=api_key)

    def test(self, prompt, model_name, structured_output=None):

        tools = None
        tool_choices = None

        if structured_output:
            if not isinstance(structured_output, str) and not isinstance(
                structured_output, dict
            ):
                structured_output = eval(structured_output.schema_json())
                remove_titles(structured_output)
            tools = [
                {
                    "name": "problem",
                    "description": "Create a new problem from the given problem",
                    "input_schema": structured_output,
                }
            ]
            tool_choices = {"type": "tool", "name": "problem"}

        response = self.client.messages.create(
            model=model_name,
            max_tokens=1024,
            system=prompt[0]["content"],
            messages=prompt[1:],
            temperature=0,
            **({"tools": tools} if tools else {}),
            **({"tool_choice": tool_choices} if tool_choices else {}),
        )
        response_data = response.content[0].to_dict()
        return response_data.get("input", response_data.get("text"))

    def create_batch_file(
        self, message_list, model_name, batch_file, id_list=None, structured_output=None
    ):

        if id_list is None:
            id_list = [i for i in range(len(message_list))]

        if structured_output and not isinstance(structured_output, str):
            structured_output = eval(structured_output.schema_json())
            remove_titles(structured_output)

        request_list = []
        for id, message in zip(id_list, message_list):
            request = {
                "id": id,
                "message": message,
                "model_name": model_name,
                "structured_output": structured_output,
            }
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
                request["message"], request["model_name"], request["structured_output"]
            )
            response_list.append({"id": request["id"], "response": response})
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

        # 1. 배치 파일 열기
        with open(batch_file, "r", encoding="utf-8") as file:
            request_list = [json.loads(line) for line in file]

        # 2. 배치 크기마다 API 호출
        batch_id_list = []
        for i in range(0, len(request_list), batch_size):
            requests = []
            for j in range(batch_size):
                if i + j >= len(request_list):
                    break

                tools = None
                tool_choices = None
                if request_list[i + j]["structured_output"]:
                    tools = [
                        {
                            "name": "problem",
                            "description": "Create a new problem from the given problem",
                            "input_schema": request_list[i + j]["structured_output"],
                        }
                    ]
                    tool_choices = {"type": "tool", "name": "problem"}

                requests.append(
                    Request(
                        custom_id=str(request_list[i + j]["id"]),
                        params=MessageCreateParamsNonStreaming(
                            model=request_list[i + j]["model_name"],
                            max_tokens=1024,
                            system=request_list[i + j]["message"][0]["content"],
                            messages=request_list[i + j]["message"][1:],
                            **({"tools": tools} if tools else {}),
                            **({"tool_choice": tool_choices} if tool_choices else {}),
                        ),
                    )
                )
            batch_job = self.client.beta.messages.batches.create(requests=requests)
            batch_id_list.append(batch_job.id)

        for batch_id in batch_id_list:
            print(f"batch id: {batch_id}")
        return batch_id_list

    def retrieve_batch(self, output_file, batch_id):
        batch_status = self.client.beta.messages.batches.retrieve(
            batch_id
        ).processing_status
        if batch_status != "ended":
            return False

        with open(output_file, "w", encoding="utf-8") as file:
            for response in self.client.beta.messages.batches.results(batch_id):
                response_data = response.result.message.content[0].to_dict()
                result = {
                    "id": response.custom_id,
                    "response": (
                        response_data.get("input", response_data.get("text"))
                        if response.result.type == "succeeded"
                        else None
                    ),
                }
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
        return True
