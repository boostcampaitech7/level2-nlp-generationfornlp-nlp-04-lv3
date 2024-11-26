import os
import json
import pandas as pd
from dotenv import load_dotenv
from syn_data.api.utils import Problems

from syn_data.api.openai_api import OpenAIApi
from syn_data.api.claude_api import ClaudeApi
from syn_data.prompt_builder import SynDataGenPromptBuilder


class SynDataGenerator:

    def __init__(self, openai_key, claude_key, data_file, save_dir):

        self.apis = {
            "OpenAI": OpenAIApi(openai_key) if openai_key else None,
            "Claude": ClaudeApi(claude_key) if claude_key else None,
        }
        self.prompt_builder = SynDataGenPromptBuilder()
        self.data_file = data_file
        self.running_batch = dict()
        self.save_dir = save_dir

    def test(self, instruction, idx, api_provider, model_name, structured_output):

        # 1. 프롬프트 생성
        id_list, prompt_list = self.prompt_builder.create_prompt_list(
            instruction, self.data_file
        )
        prompt = prompt_list[idx]

        # 2. API 호출
        response = self.apis[api_provider].test(prompt, model_name, structured_output)
        return prompt, response

    def run(
        self,
        instruction,
        job,
        api_provider,
        model_name,
        structured_output,
        batch_size,
        type,
    ):
        # 1. 프롬프트 생성
        id_list, prompt_list = self.prompt_builder.create_prompt_list(
            instruction, self.data_file
        )

        # 2. 프롬프트 파일 생성
        batch_file = f"{self.save_dir}/batch/{os.path.splitext(os.path.basename(self.data_file))[0]}_{api_provider}_{model_name}_{job}.jsonl"
        self.apis[api_provider].create_batch_file(
            id_list=id_list,
            message_list=prompt_list,
            model_name=model_name,
            batch_file=batch_file,
            structured_output=structured_output,
        )

        # 3. API 호출
        if type == "batch":
            batch_id_list = self.apis[api_provider].call_batch(batch_file, batch_size)
            for idx, batch_id in enumerate(batch_id_list):
                self.running_batch[batch_id] = (
                    api_provider,
                    f"{os.path.splitext(batch_file)[0]}_output{idx}.jsonl",
                )
        else:
            self.apis[api_provider].call(batch_file, batch_size)

    def test_aug(self, instruction, idx, api_provider, model_name):
        return self.test(
            instruction, idx, api_provider, model_name, structured_output=Problems
        )

    def test_etc(self, instruction, idx, api_provider, model_name):
        return self.test(
            instruction, idx, api_provider, model_name, structured_output=None
        )

    def aug(self, instruction, api_provider, model_name, batch_size=100, type="batch"):
        self.run(
            instruction,
            "aug",
            api_provider,
            model_name,
            structured_output=Problems,
            batch_size=batch_size,
            type=type,
        )

    def cot(self, instruction, api_provider, model_name, batch_size=100, type="batch"):
        self.run(
            instruction,
            "cot",
            api_provider,
            model_name,
            structured_output=None,
            batch_size=batch_size,
            type=type,
        )

    def compare(
        self, instruction, api_provider, model_name, batch_size=100, type="batch"
    ):
        self.run(
            instruction,
            "compare",
            api_provider,
            model_name,
            structured_output=None,
            batch_size=batch_size,
            type=type,
        )

    def format_batch_to_csv(self, batch_output_file, save_file):
        # batch로 추출된 걸 csv formatting함
        result_list = []
        with open(batch_output_file, "r", encoding="utf-8") as file:
            for line in file:
                result = json.loads(line.strip())
                result_list.append(result)

        id_list, paragraph_list, problems_list, question_plus_list = [], [], [], []
        for result in result_list:
            id_list.append(result["id"])
            paragraph_list.append(result["response"].get("paragraph", ""))
            question_plus_list.append(result["response"].get("note", ""))
            problems_list.append(
                {
                    "question": result["response"]["question"],
                    "choices": result["response"]["choices"],
                    "answer": result["response"]["answer"],
                }
            )
        df = pd.DataFrame()
        df["id"] = id_list
        df["paragraph"] = paragraph_list
        df["problems"] = problems_list
        df["question_plus"] = question_plus_list
        df.to_csv(save_file, index=False)

    def retrieve_batchs(self):
        completed_batch_list = []
        for batch_id, (api_provider, batch_output_file) in list(
            self.running_batch.items()
        ):
            if self.apis[api_provider].retrieve_batch(batch_output_file, batch_id):
                self.running_batch.pop(batch_id)
                completed_batch_list.append(batch_id)
                csv_file = f"{self.save_dir}/syn_data/{os.path.splitext(os.path.basename(batch_output_file))[0]}.csv"
                self.format_batch_to_csv(batch_output_file, csv_file)
        return completed_batch_list
