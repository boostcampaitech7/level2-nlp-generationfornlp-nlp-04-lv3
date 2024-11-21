from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import pandas as pd
import ast
from tqdm import tqdm
import re
import tiktoken  # 토크나이저 추가
import os

model_name = "qwen2.5:14b"
# 최대 입력 토큰 수 설정
MAX_TOKENS = 5000  # 원하는 최대 토큰 수로 설정

# 토크나이저 초기화 (모델에 맞는 토크나이저 사용)
# 여기서는 예시로 GPT-3의 tiktoken 사용
tokenizer = tiktoken.get_encoding("gpt2")  # Ollama 모델에 맞는 인코딩 사용


def count_tokens(text):
    return len(tokenizer.encode(text))


def truncate_paragraph(paragraph, max_tokens, buffer_tokens=500):
    """
    paragraph를 최대 토큰 수에 맞게 잘라내는 함수.
    buffer_tokens는 프롬프트의 다른 부분에서 사용되는 토큰 수를 고려한 여유분입니다.
    """
    tokens = tokenizer.encode(paragraph)
    if len(tokens) > (max_tokens - buffer_tokens):
        truncated_tokens = tokens[: (max_tokens - buffer_tokens)]
        return tokenizer.decode(truncated_tokens)
    return paragraph


# Ollama 모델을 초기화합니다.
def initialize_llm(model_name):
    return OllamaLLM(model=model_name, base_url="http://127.0.0.1:11434")


# 결과 저장을 위한 클래스 정의
class ResultSaver:
    def __init__(self, file_path, save_every=5):
        self.file_path = file_path
        self.save_every = save_every
        self.results = []
        self.counter = 0
        # 기존 결과 불러오기
        if os.path.exists(self.file_path):
            self.existing_df = pd.read_csv(self.file_path)
        else:
            self.existing_df = pd.DataFrame()

    def add_result(self, result):
        self.results.append(result)
        self.counter += 1
        if self.counter >= self.save_every:
            self.save_results()
            self.counter = 0
            self.results = []

    def save_results(self):
        results_df = pd.DataFrame(self.results)
        if not results_df.empty:
            if not os.path.exists(self.file_path):
                # 파일이 없으면 헤더와 함께 저장
                results_df.to_csv(self.file_path, index=False)
            else:
                # 파일이 있으면 기존 데이터와 병합하여 저장
                existing_df = pd.read_csv(self.file_path)
                combined_df = pd.concat([existing_df, results_df], ignore_index=True)
                combined_df.drop_duplicates(subset="id", keep="last", inplace=True)
                combined_df.to_csv(self.file_path, index=False)
            # 기존 데이터 업데이트
            self.existing_df = pd.read_csv(self.file_path)


# 정답 예측을 위한 함수 정의
def process_row(row, llm):
    try:
        # 'problems' 컬럼을 딕셔너리로 변환
        problems = ast.literal_eval(row["problems"])
    except Exception as e:
        result = {
            "id": row["id"],
            "predicted_answer": None,
            "actual_answer": None,
            "error": f"Parsing error: {e}",
        }
        return result

    question = problems.get("question", "")
    choices_list = problems.get("choices", [])
    actual_answer = problems.get("answer", None)  # 실제 정답 번호

    # 선택지를 번호와 함께 문자열로 변환
    choices = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices_list)])

    # paragraph 길이 조절
    paragraph = row["paragraph"]
    truncated_paragraph = truncate_paragraph(paragraph, MAX_TOKENS)

    # 프롬프트 생성
    prompt = f"""
paragraph:
{truncated_paragraph}

question:
{question}

choices:
{choices}
Just print out plain single number. Don't add any Explanation.
answer number:
""".strip()

    # 프롬프트 토큰 수 확인 및 추가 조정 (필요 시)
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > MAX_TOKENS:
        # 추가로 잘라낼 필요가 있는 경우
        excess = prompt_tokens - MAX_TOKENS
        truncated_paragraph = truncate_paragraph(
            paragraph, MAX_TOKENS, buffer_tokens=500 + excess
        )
        prompt = f"""
paragraph:
{truncated_paragraph}

question:
{question}

choices:
{choices}

Just print out plain single number. Don't add any Explanation.
answer number:
""".strip()

    try:
        # 스트리밍을 사용하여 실시간으로 출력 처리
        answer_number = None
        response = ""
        for token in llm.invoke(prompt, stream=True):
            response += token
            print(token, end="", flush=True)  # 실시간 출력 확인 (옵션)
            if token.strip().isdigit():
                answer_number = int(token.strip())
                break  # 숫자를 얻으면 생성을 멈춤
    except Exception as e:
        result = {
            "id": row["id"],
            "predicted_answer": None,
            "actual_answer": actual_answer,
            "error": f"LLM error: {e}",
        }
        return result

    if answer_number is None:
        # 숫자를 얻지 못한 경우 전체 응답에서 숫자 추출 시도
        answer_number_str = "".join(filter(str.isdigit, response))
        if answer_number_str:
            answer_number = int(answer_number_str)
        else:
            answer_number = None

    result = {
        "id": row["id"],
        "predicted_answer": answer_number,
        "actual_answer": actual_answer,
        "error": None,
    }
    return result


def main():
    # CSV 데이터 로드
    data = pd.read_csv(
        "/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/test.csv"
    )

    # 결과 파일 경로
    result_file_path = "/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/output/nemotron_predicted_test.csv"

    # 결과 저장기 초기화
    result_saver = ResultSaver(result_file_path, save_every=5)

    # 이미 처리된 데이터 로드
    if os.path.exists(result_file_path):
        existing_results = pd.read_csv(result_file_path)
        # 에러가 있는 행만 선택
        error_ids = existing_results[existing_results["error"].notnull()]["id"].tolist()
        # 에러가 없는 행의 ID
        success_ids = existing_results[existing_results["error"].isnull()][
            "id"
        ].tolist()
    else:
        error_ids = []
        success_ids = []

    # 에러가 발생한 행만 재처리
    if error_ids:
        data_to_process = data[data["id"].isin(error_ids)]
    else:
        # 처음 실행하는 경우 전체 데이터 처리
        data_to_process = data

    # 이미 성공적으로 처리된 ID는 제외
    data_to_process = data_to_process[~data_to_process["id"].isin(success_ids)]

    # Ollama 모델 초기화
    llm = initialize_llm(model_name)

    # 데이터 처리 (병렬 처리 제거, 순차적으로 처리)
    for index, row in tqdm(
        data_to_process.iterrows(),
        total=data_to_process.shape[0],
        desc="Processing Rows",
    ):
        result = process_row(row, llm)
        result_saver.add_result(result)

    # 남은 결과 저장
    if result_saver.results:
        result_saver.save_results()
        result_saver.results = []

    # 모든 결과가 저장되었으므로, 결과 파일 로드
    results_df = pd.read_csv(result_file_path)

    # 원본 데이터와 결과 병합 (id 기준)
    merged_data = pd.merge(data, results_df, on="id", how="left")

    # 에러가 있는 행 확인 (선택 사항)
    errors = merged_data[merged_data["error"].notnull()]
    if not errors.empty:
        print(f"\n에러가 발생한 행 수: {errors.shape[0]}")
        print(errors[["id", "error"]])

    # 정확도 계산 (정답과 예측이 모두 존재하는 경우에만 계산)
    valid_cases = merged_data.dropna(subset=["predicted_answer", "actual_answer"])
    correct = valid_cases[
        valid_cases["predicted_answer"] == valid_cases["actual_answer"]
    ].shape[0]
    incorrect = valid_cases[
        valid_cases["predicted_answer"] != valid_cases["actual_answer"]
    ].shape[0]
    total = valid_cases.shape[0]
    accuracy = (correct / total) * 100 if total > 0 else 0

    # 결과 출력
    print(f"\n총 처리된 문제 수: {total}")
    print(f"맞춘 개수: {correct}")
    print(f"틀린 개수: {incorrect}")
    print(f"정확도: {accuracy:.2f}%")

    # 정제된 데이터 저장
    merged_data.to_csv(result_file_path, index=False)
    print(f"예측된 정답이 {result_file_path} 파일에 저장되었습니다.")


if __name__ == "__main__":
    main()
