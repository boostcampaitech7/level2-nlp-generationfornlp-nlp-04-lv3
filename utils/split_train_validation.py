import pandas as pd
from sklearn.model_selection import train_test_split
import ast
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")

# 1. Balanced된 데이터 로드
df = pd.read_csv(f"{ROOT_DIR}/data/default/train.csv")


# 2. 'problems' 열을 파싱하여 'answer'와 'num_choices' 추출
def extract_num_choices(problems_str):
    try:
        problems = ast.literal_eval(problems_str)
        return len(problems["choices"])
    except:
        return None


def extract_answer(problems_str):
    try:
        problems = ast.literal_eval(problems_str)
        return problems["answer"]
    except:
        return None


# 'num_choices'와 'answer' 열 생성
df["num_choices"] = df["problems"].apply(extract_num_choices)
df["answer"] = df["problems"].apply(extract_answer)

# 3. 층화 추출을 위한 결합 열 생성
df["stratify_col"] = df["num_choices"].astype(str) + "_" + df["answer"].astype(str)

# 4. 데이터 분할 (예: 80% 학습용, 20% 검증용)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,  # 검증용 데이터 비율 (20%)
    random_state=42,  # 재현성을 위한 랜덤 시드
    stratify=df["stratify_col"],  # 층화 추출 기준
)

# 5. 불필요한 열 제거
train_df = train_df.drop(columns=["num_choices", "answer", "stratify_col"])
val_df = val_df.drop(columns=["num_choices", "answer", "stratify_col"])

# 6. 분할된 데이터 저장
train_df.to_csv(f"{ROOT_DIR}/data/default/flatten_answers_train.csv", index=False)
val_df.to_csv(f"{ROOT_DIR}/data/default/flatten_answers_validation.csv", index=False)


# 7. 분할 후 분포 확인
def print_distribution(df, dataset_name):
    distribution = defaultdict(int)
    for problems_str in df["problems"]:
        try:
            problems = ast.literal_eval(problems_str)
            answer = problems["answer"]
            distribution[answer] += 1
        except:
            pass
    print(f"\n{dataset_name} 답의 분포:")
    for answer, count in sorted(distribution.items()):
        print(f"Answer {answer}: {count}")


print_distribution(train_df, "학습용 데이터")
print_distribution(val_df, "검증용 데이터")
