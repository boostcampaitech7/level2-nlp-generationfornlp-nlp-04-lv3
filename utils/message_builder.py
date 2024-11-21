import pandas as pd
from ast import literal_eval


class MessageBuilder:

    def __init__(self, api_type="openai"):
        self.api_type = api_type

        self.PROMPT_NO_QUESTION_PLUS = """지문:
                {paragraph}

                질문:
                {question}

                선택지:
                {choices}

                1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
                정답: {answer}"""

        self.PROMPT_QUESTION_PLUS = """지문:
                {paragraph}

                질문:
                {question}

                <보기>:
                {question_plus}

                선택지:
                {choices}

                1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
                정답: {answer}"""

    def create_user_message(self, row):
        problem = literal_eval(row["problems"])
        choices_string = "\n".join(
            [f"{idx + 1} - {choice}" for idx, choice in enumerate(problem["choices"])]
        )

        if not pd.isna(row["question_plus"]):
            user_message = self.PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=problem["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
                answer=problem["answer"],
            )
        else:
            user_message = self.PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=problem["question"],
                choices=choices_string,
                answer=problem["answer"],
            )

        return user_message

    def create_message_list(self, file_path, system_message=""):

        df = pd.read_csv(file_path)

        id_list, message_list = [], []

        for index, row in df.iterrows():
            id_list.append(row["id"])
            user_message = self.create_user_message(row)

            # OpenAI API 요청 메시지
            if self.api_type == "openai":
                message = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ]
            # Claude API 요청 메시지
            elif self.api_type == "claude":
                message = [{"role": "user", "content": user_message}]
            # 다른 API 요청시 에러 발생
            else:
                raise ValueError(f"Unsupported API type: {self.api_type}")

            message_list.append(message)

        return id_list, message_list


if __name__ == "__main__":
    message_builder = MessageBuilder("claude")
    id_list, message_list = message_builder.create_message_list(
        system_message="아래 문제 정답이 지문 안에 있는지에 대한 여부 알려주세요.",
        file_path="/data/ephemeral/home/ms/level2-nlp-generationfornlp-nlp-04-lv3/data/default/train.csv",
    )
    print(message_list[0])
