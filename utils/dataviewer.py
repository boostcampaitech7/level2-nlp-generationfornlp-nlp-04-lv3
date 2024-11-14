import streamlit as st
import pandas as pd
import ast


def view_data(df):
    select_options = [
        f'{row["id"]}: {ast.literal_eval(row["problems"])["question"]}'
        for index, row in df.iterrows()
    ]
    selected_question = st.selectbox("질문을 선택하세요", select_options)
    selected_id = selected_question.split(":")[0]

    selected_row = df[df["id"] == selected_id].squeeze()

    view_row(selected_row)


def view_row(row):
    paragraph = row["paragraph"]
    problems = ast.literal_eval(row["problems"])
    question = problems["question"]
    choices = problems["choices"]

    st.markdown(
        f"""
        <div style='background-color: #f7f7ff; border-radius: 10px; padding: 20px;'>
        <div style='font-weight: bold; font-size: 20px; margin-bottom: 20px;'>지문: </div>
        {paragraph}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        f"<div style='font-weight: bold; font-size: 20px; margin-bottom: 20px;'>질문: {question} </div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='font-weight: bold; font-size: 20px; margin-bottom: 20px;'>선지: </div>",
        unsafe_allow_html=True,
    )
    for i, choice in enumerate(choices):
        st.markdown(f"{i+1}: {choice}", unsafe_allow_html=True)


def data_page(df):
    train_tab, test_tab = st.tabs(["Train", "Test"])

    with train_tab:
        view_data(df["train"])

    with test_tab:
        view_data(df["test"])


def main(dir_path):
    # 화면 레이아웃 설정
    st.set_page_config(layout="wide", page_title="SEVEN ELEVEN ODQA Data Viewer V2.0.0")

    # 우선 default 데이터셋만 볼 수 있게 함
    df_train = pd.read_csv(f"{dir_path}/train.csv")
    df_test = pd.read_csv(f"{dir_path}/test.csv")
    df = {"train": df_train, "test": df_test}

    st.sidebar.title("페이지 선택")

    page = st.sidebar.selectbox("Choose a page", ("Data Page"))

    if page == "Data Page":
        data_page(df)


if __name__ == "__main__":
    data_path = "/data/ephemeral/home/data"
    main(data_path)
