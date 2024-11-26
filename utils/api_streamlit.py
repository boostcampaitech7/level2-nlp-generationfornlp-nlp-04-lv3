import os
import json
import time
from dotenv import load_dotenv
import ast
import pytz
from syn_data.syn_data_gen import SynDataGenerator
import pandas as pd
import streamlit as st
from datetime import datetime


aug_prompt = """당신은 수능 문제 출제위원입니다. 문제의 지문과 질문에 사용된 핵심 개념이나 내용을 바탕으로 새로운 수능형 문제를 작성하시오. 작성할 문제는 다음 조건을 충족해야 합니다:
1. 핵심 개념 연계: 지문과 질문은 기존 문제의 핵심 개념과 관련되되, 내용은 완전히 새롭게 구성합니다.
2. 지문 길이 유지: 새로운 지문의 길이는 기존 문제의 지문 길이와 무조건 동일해야 합니다.
3. 선택지 설계: 선택지는 5개로 구성하며, 하나만 정답이 되도록 논리적이고 치밀하게 설계합니다.
4. 수능형 스타일: 지문과 문제는 수능 시험의 형식과 논리 구조에 맞춰 작성합니다."""

cot_prompt = """당신은 수능 문제 해설에 능숙한 "일타강사"입니다. 제공된 수능형 문제를 읽고, 다음 조건에 맞는 풀이 과정을 작성하세요.
1. 선택지 검토: 각 선택지를 하나씩 분석하며, 정답 여부를 판단하는 명확한 근거를 구체적이고 자세하게 제시합니다.
2. 선 근거 후 결과: 반드시 근거를 제시한 후, 해당 선택지의 정답 여부를 제시합니다.
3. 지문 및 <보기> 활용: 문제의 지문 또는 <보기>를 적극적으로 활용하여 풀이 과정을 논리적으로 전개합니다."""


def read_history_file(history_file):
    if not os.path.exists(history_file):
        with open(history_file, "w") as file:
            file.write("")  # 빈 JSON 객체로 파일 생성

    history = dict()
    with open(history_file, "r+") as file:
        history = {
            line_data["batch_id"]: (
                line_data["api_provider"],
                line_data["batch_output_file"],
            )
            for line in file
            if line.strip() and (line_data := json.loads(line))["status"] != "completed"
        }
    return history


def write_history_file(history_file, api_type, batch_id, batch_output_file, model_name):
    new_record = {
        "start_time": datetime.now(pytz.timezone("Asia/Seoul")).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "api_provider": api_type,
        "status": "running",  # completed
        "batch_id": batch_id,
        "batch_output_file": batch_output_file,
        "model_name": model_name,
    }
    with open(history_file, "a") as file:
        file.write(json.dumps(new_record, ensure_ascii=False) + "\n")


def update_history_file(history_file, batch_id_list):
    update_records = []
    with open(history_file, "r+") as file:
        for line in file:
            if line.strip():
                record = json.loads(line.strip())
                if record["batch_id"] in batch_id_list:
                    record["status"] = "completed"
                update_records.append(record)

    with open(history_file, "w") as file:
        for record in update_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def view_data(df):
    select_options = [
        f'{row["id"]}: {ast.literal_eval(row["problems"])["question"]}'
        for index, row in df.iterrows()
    ]
    selected_question = st.selectbox("질문을 선택하세요", select_options)
    selected_id = selected_question.split(":")[0]

    selected_row = df[df["id"] == selected_id].squeeze()

    view_problem(selected_row)
    return df.index[df["id"] == selected_id][0]


def view_problem(row):
    if isinstance(row, pd.Series):
        row = row.fillna(value="").to_dict()
        row.update(ast.literal_eval(row.pop("problems")))

    if isinstance(row, dict):
        row["question_plus"] = row.get("note", None)

    if row["paragraph"] and row["paragraph"] != "":
        st.markdown(
            f"""
            <div style='background-color: #f7f7ff; border-radius: 10px; padding: 20px;'>
            <div style='font-weight: bold; font-size: 20px; margin-bottom: 20px;'>지문: </div>
            {row['paragraph']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if row["question_plus"] and row["question_plus"] != "":
        st.markdown(
            f"""
            <br>
            <div style='background-color: #f7f7fe; border-radius: 8px; padding: 20px;'>
            <div style='font-weight: bold; font-size: 17px; margin-bottom: 17px;'><보기></div>
            {row['question_plus']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        f"<div style='font-weight: bold; font-size: 20px; margin-bottom: 20px;'>질문: {row['question']} </div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='font-weight: bold; font-size: 20px; margin-bottom: 20px;'>선지: </div>",
        unsafe_allow_html=True,
    )
    for i, choice in enumerate(row["choices"]):
        st.markdown(f"{i+1}: {choice}", unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-weight: bold; font-size: 20px; margin-bottom: 20px;'>정답: {row['answer']}</div>",
        unsafe_allow_html=True,
    )


def main(api_key, data_dir, history_file):

    st.set_page_config(layout="wide")
    syn_data_gen = SynDataGenerator(
        api_key["openai"], api_key["claude"], data_file=None, save_dir=data_dir
    )
    syn_data_gen.running_batch = read_history_file(history_file)

    # 사이드바 선택 항목 부분
    model_options_dict = {
        "OpenAI": ["gpt-4o-mini", "gpt-4o"],
        "Claude": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
    }
    st.sidebar.title("선택 항목")
    api_type = st.sidebar.radio("API 제공자", model_options_dict.keys())
    model_name = st.sidebar.radio("모델 이름", model_options_dict[api_type])
    data_file = st.sidebar.radio(
        "데이터 파일", [file for file in os.listdir(data_dir) if file.endswith(".csv")]
    )
    # 배치 작업 완료된 API 요청 처리
    if st.sidebar.button("Retrieve Batch API"):
        completed_batch_list = syn_data_gen.retrieve_batchs()
        if completed_batch_list:
            update_history_file(history_file, completed_batch_list)

        placeholder = st.sidebar.empty()
        placeholder.write("Finish Retrieve Batch API")
        time.sleep(1)
        placeholder.empty()

    # 데이터 로드
    syn_data_gen.data_file = os.path.join(data_dir, data_file)
    df = pd.read_csv(os.path.join(data_dir, data_file))

    # 데이터 시각화
    idx = view_data(df)

    # API 호출
    on = st.toggle("OFF (CoT 데이터 생성) / ON (합성 데이터 생성)", value=True)
    prompt, task = (
        (aug_prompt, "합성 데이터 생성") if on else (cot_prompt, "CoT 데이터 생성")
    )
    prompt = st.text_area(f"{task} 프롬프트", height=200, value=prompt)

    if st.button("Call API"):
        if on:
            _, result = syn_data_gen.test_aug(prompt, idx, api_type, model_name)
            view_problem(result)
        else:
            _, result = syn_data_gen.test_etc(prompt, idx, api_type, model_name)
            st.markdown(
                f"""
                <div style='background-color: #f7f7ff; border-radius: 10px; padding: 20px;'>
                <div style='font-weight: bold; font-size: 20px; margin-bottom: 20px;'>문제풀이: </div>
                {result}
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("Create Batch API"):
        if on:
            syn_data_gen.aug(prompt, api_type, model_name)
            st.write("파일 전체 제출 완료")
            batch_id = next(reversed(syn_data_gen.running_batch))
            _, batch_output_file = syn_data_gen.running_batch[batch_id]
            write_history_file(
                history_file, api_type, batch_id, batch_output_file, model_name
            )
        else:
            st.write("CoT 배치 기능은 아직 구현되지 않았습니다.")
            # syn_data_gen.cot(prompt, api_type, model_name)


if __name__ == "__main__":
    load_dotenv()
    api_key = {
        "openai": os.getenv("OPENAI_KEY"),
        "gemini": os.getenv("GEMINI_KEY"),
        "claude": os.getenv("CLAUDE_KEY"),
    }
    data_dir = (
        "/data/ephemeral/home/gj/level2-nlp-generationfornlp-nlp-04-lv3/_data/test"
    )
    main(api_key, data_dir, history_file=os.path.join(data_dir, "history.jsonl"))
