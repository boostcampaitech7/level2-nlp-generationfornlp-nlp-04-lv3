import pandas as pd
import ast
import random
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")

# 1. train.csv 파일 로드
df = pd.read_csv(f"{ROOT_DIR}/data/default/train.csv")

# 2. 'problems' 열을 딕셔너리로 변환하여 새로운 열 'question_plus'에 저장
df["question_plus"] = df["problems"].apply(ast.literal_eval)


# 3. 초기 답의 분포 확인
def print_initial_distribution(df):
    distribution = defaultdict(lambda: defaultdict(int))
    for q_plus in df["question_plus"]:
        num_choices = len(q_plus["choices"])
        answer = q_plus["answer"]
        distribution[num_choices][answer] += 1
    print("초기 답의 분포:")
    for num_choices in sorted(distribution.keys()):
        print(f"\n선택지 {num_choices}개인 문제의 분포:")
        counts = pd.Series(distribution[num_choices]).sort_index()
        print(counts)


print_initial_distribution(df)

# 4. 선택지 개수별 목표 분포 계산
choice_counts = df["question_plus"].apply(lambda x: len(x["choices"])).value_counts()

# 목표 분포 저장: {num_choices: {position: target_count}}
target_counts = {}
for num_choices, count in choice_counts.items():
    base = count // num_choices
    remainder = count % num_choices
    target = {i: base for i in range(1, num_choices + 1)}
    for i in range(1, remainder + 1):
        target[i] += 1
    target_counts[num_choices] = target

# 5. 선택지 개수별 현재 할당된 답 수 초기화
current_counts = {num_choices: defaultdict(int) for num_choices in target_counts.keys()}

# 6. 균일한 분포를 위해 수정된 선택지와 답을 저장할 리스트 초기화
balanced_choices = []
balanced_answers = []

# 7. 각 질문에 대해 균일하게 답을 할당
for idx, row in df.iterrows():
    q_plus = row["question_plus"]
    choices = q_plus["choices"]
    num_choices = len(choices)
    correct_answer_index = q_plus["answer"] - 1  # 0-based index
    correct_choice = choices[correct_answer_index]

    # 가능한 선택지 위치 중 목표를 달성하지 않은 위치 찾기
    available_positions = [
        pos
        for pos in range(1, num_choices + 1)
        if current_counts[num_choices][pos] < target_counts[num_choices][pos]
    ]

    if not available_positions:
        # 모든 위치가 목표를 달성했다면, 전체 선택지 중 최소 할당 위치 선택
        min_count = min(current_counts[num_choices].values())
        candidate_positions = [
            pos
            for pos in range(1, num_choices + 1)
            if current_counts[num_choices][pos] == min_count
        ]
        target_position = random.choice(candidate_positions)
    else:
        # 현재 가장 적게 할당된 위치 선택
        min_count = min(
            [current_counts[num_choices][pos] for pos in available_positions]
        )
        candidate_positions = [
            pos
            for pos in available_positions
            if current_counts[num_choices][pos] == min_count
        ]
        # 랜덤하게 후보 중 하나 선택
        target_position = random.choice(candidate_positions)

    # 정답 위치에 할당 및 카운트 업데이트
    current_counts[num_choices][target_position] += 1

    # 나머지 선택지 준비 및 무작위 섞기
    other_choices = choices[:correct_answer_index] + choices[correct_answer_index + 1 :]
    random.shuffle(other_choices)

    # 새로운 선택지 리스트 생성
    new_choices = other_choices.copy()
    new_choices.insert(target_position - 1, correct_choice)

    # 수정된 선택지와 답 리스트에 추가
    balanced_choices.append(new_choices)
    balanced_answers.append(target_position)


# 8. 수정된 선택지와 답을 기존 'problems' 열에 반영
def create_balanced_problems(row, new_choices, new_answer):
    q_plus = row["question_plus"]
    updated_q_plus = {
        "question": q_plus["question"],
        "choices": new_choices,
        "answer": new_answer,
    }
    # 문자열 형태로 반환 (single quotes 유지)
    return str(updated_q_plus)


df["problems"] = df.apply(
    lambda row: create_balanced_problems(
        row, balanced_choices[row.name], balanced_answers[row.name]
    ),
    axis=1,
)


# 9. 수정 후 답의 분포 확인
def print_final_distribution(df):
    distribution = defaultdict(lambda: defaultdict(int))
    for q_plus_str in df["problems"]:
        q_plus = ast.literal_eval(q_plus_str)
        num_choices = len(q_plus["choices"])
        answer = q_plus["answer"]
        distribution[num_choices][answer] += 1
    print("\n수정 후 답의 분포:")
    for num_choices in sorted(distribution.keys()):
        print(f"\n선택지 {num_choices}개인 문제의 분포:")
        counts = pd.Series(distribution[num_choices]).sort_index()
        print(counts)


print_final_distribution(df)

# 10. 변경 사항을 새로운 파일로 저장 (필요 시)
df.to_csv(f"{ROOT_DIR}/data/default/flatten_answers_train.csv", index=False)
