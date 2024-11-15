from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import pandas as pd
import ast
from tqdm import tqdm
import concurrent.futures
import re
import tiktoken  # 토크나이저 추가

# 최대 입력 토큰 수 설정
MAX_TOKENS = 3000  # 원하는 최대 토큰 수로 설정

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


# 함수 내에서 Ollama 모델을 초기화하여 각 스레드에서 독립적으로 사용할 수 있도록 합니다.
def initialize_llm():
    return OllamaLLM(model="m/gemma2:27b-max")


# 정답 예측을 위한 함수 정의
def process_row(row):
    try:
        # 'problems' 컬럼을 딕셔너리로 변환
        problems = ast.literal_eval(row["problems"])
    except Exception as e:
        return {
            "id": row["id"],
            "predicted_answer": None,
            "actual_answer": None,
            "error": f"Parsing error: {e}",
        }

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
다음은 주어진 글과 문제입니다. 주어진 선택지 중에서 가장 적절한 정답의 번호를 숫자로만 답해주세요. 출력 형식을 엄격히 지키세요.

글:
{truncated_paragraph}

문제:
{question}

선택지:
{choices}

정답 번호:
""".strip()

    # 프롬프트 토큰 수 확인 및 추가 조정 (필요 시)
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > MAX_TOKENS:
        # 추가로 잘라낼 필요가 있는 경우 (예: 문제나 선택지도 길다면 추가 조정)
        # 여기서는 간단히 paragraph를 더 잘라내는 예시를 들었습니다.
        excess = prompt_tokens - MAX_TOKENS
        # paragraph 다시 잘라내기
        truncated_paragraph = truncate_paragraph(
            paragraph, MAX_TOKENS, buffer_tokens=500 + excess
        )
        prompt = f"""
다음은 주어진 글과 문제입니다. 주어진 선택지 중에서 가장 적절한 정답의 번호를 숫자로만 답해주세요. 출력 형식을 엄격히 지키세요.

글:
{truncated_paragraph}

문제:
{question}

선택지:
{choices}

정답 번호:
""".strip()

    try:
        # Ollama 모델 초기화
        llm = initialize_llm()
        # 모델에 프롬프트 전달하여 응답 받기
        response = llm(prompt)
        # 응답에서 숫자 추출
        answer_number_str = "".join(filter(str.isdigit, response))
        if answer_number_str:
            answer_number = int(answer_number_str)
        else:
            answer_number = None
    except Exception as e:
        return {
            "id": row["id"],
            "predicted_answer": None,
            "actual_answer": actual_answer,
            "error": f"LLM error: {e}",
        }

    return {
        "id": row["id"],
        "predicted_answer": answer_number,
        "actual_answer": actual_answer,
        "error": None,
    }


def main():
    # CSV 데이터 로드
    data = pd.read_csv(
        "/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/test.csv"
    )

    # 결과를 저장할 리스트 초기화
    results = []

    # ThreadPoolExecutor를 사용하여 병렬 처리
    max_workers = 10  # 스레드 수 조정 (시스템 사양과 API 제한에 따라 조정)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # tqdm을 사용하여 진행 상황 표시
        futures = {
            executor.submit(process_row, row): row["id"]
            for index, row in data.iterrows()
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing Rows",
        ):
            result = future.result()
            results.append(result)

    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)

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
    merged_data.to_csv("./data/predicted_test.csv", index=False)
    print("예측된 정답이 predicted_test.csv 파일에 저장되었습니다.")


if __name__ == "__main__":
    main()
