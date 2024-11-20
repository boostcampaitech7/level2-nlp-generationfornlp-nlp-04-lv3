import pandas as pd
from ast import literal_eval


class SynDataGenPromptBuilder:

    def __init__(self):
        self.format = {
            "paragraph": "\n지문:\n{paragraph}\n",
            "question": "\n질문:\n{question}\n",
            "question_plus": "\n<보기>:\n{question_plus}\n",
            "choices": "\n선택지:\n{choices}\n",
            "sovling": "\n문제풀이:\n{solving}\n",
            "answer": "\n정답: {answer}",
        }

    def create_data_sample_prompt(self, row):
        row = row.to_dict()
        problem = literal_eval(row.pop("problems"))
        row["choices"] = "\n".join(
            [f"{idx + 1}. {choice}" for idx, choice in enumerate(problem["choices"])]
        )
        row["question"] = problem["question"]
        row["answer"] = problem["answer"]

        data_sample_prompt = ""
        for col, pattern in self.format.items():
            if col in row and row[col]:
                data_sample_prompt = data_sample_prompt + pattern.format(
                    **{col: row[col]}
                )

        return data_sample_prompt

    def create_prompt_list(self, instruction, data_file):
        df = pd.read_csv(data_file)

        id_list, prompt_list = [], []
        for index, row in df.fillna(False).iterrows():
            id_list.append(row["id"])
            data_sample_prompt = self.create_data_sample_prompt(row)
            prompt = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": data_sample_prompt},
            ]
            prompt_list.append(prompt)

        return id_list, prompt_list


if __name__ == "__main__":
    message_builder = SynDataGenPromptBuilder()
    id_list, message_list = message_builder.create_prompt_list(
        instruction="아래 문제에서 사용된 개념으로 새로운 수능형 문제를 만들어주세요.",
        data_file="/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/data/uniform/all.csv",
    )
    print(message_list[0][1]["content"])
