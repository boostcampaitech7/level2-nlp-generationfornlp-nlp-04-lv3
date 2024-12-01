import json
from pydantic import BaseModel, Field
from typing import List
import pandas as pd


MODEL_COSTS = {
    "gpt-4o-mini": [0.150 / 1000000, 0.600 / 1000000],
    "claude-3-5-sonnet-20241022": [3.00 / 1000000, 15.00 / 1000000],
    "claude-3-5-haiku-20241022": [1.00 / 1000000, 5.00 / 1000000],
    "gemini-1.5-flash": [0.075 / 1000000, 0.30 / 1000000],
}


class Problems(BaseModel):
    paragraph: str = Field(description="문제를 풀기 위한 내용을 담고 있는 본문")
    question: str = Field(description="질문 또는 문제 내용")
    note: str = Field(description="문제를 해결하기 위한 추가 정보 또는 주석")
    choices: List[str] = Field(description="선택 가능한 답변 목록")
    answer: int = Field(description="정답을 나타내는 선택지의 인덱스")


def calculate_cost(model_name, input_tokens, output_tokens):
    total_cost = (
        MODEL_COSTS[model_name][0] * input_tokens
        + MODEL_COSTS[model_name][1] * output_tokens
    )
    print(f"API 비용: ${total_cost:.5f}")


def remove_titles(obj):
    if isinstance(obj, dict):
        # title 키를 삭제
        obj.pop("title", None)
        obj.pop("items", None)
        # 각 값에 대해 재귀적으로 처리
        for key, value in obj.items():
            remove_titles(value)
    elif isinstance(obj, list):
        # 리스트 내부 요소에 대해 재귀적으로 처리
        for item in obj:
            remove_titles(item)


def merge_batch(batch_output_file_list, batch_merged_file):
    data = []
    for batch_output_file in batch_output_file_list:
        with open(batch_output_file, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line.strip()))

    with open(batch_merged_file, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False, indent=4) + "\n")


def to_csv(batch_output_file, save_file, property_path):

    # load data
    data_list = []
    with open(batch_output_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data)

    # formatting to csv
    id_list, paragraph_list, q_list, plus_list = [], [], [], []
    for row in data_list:
        id_list.append(row["id"])
        if "paragraph" in row:
            paragraph_list.append(row["paragraph"])
        else:
            paragraph_list.append("")
        if "question_plus" in row:
            plus_list.append(row["question_plus"])
        else:
            plus_list.append("")
        q_list.append(
            {
                "question": row["question"],
                "choices": [
                    ele.strip()
                    for ele in re.split("\n[1-5]\.", "\n" + row["choices"])[1:]
                ],
                "answer": int(row["answer"]),
            }
        )
    df = pd.DataFrame()
    df["id"] = id_list
    df["paragraph"] = paragraph_list
    df["problems"] = q_list
    df["question_plus"] = plus_list
    df.to_csv(save_file, index=False)
