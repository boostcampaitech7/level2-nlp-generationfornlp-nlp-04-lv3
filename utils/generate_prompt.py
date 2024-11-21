# 이 파일에 있는 함수들은
# 프롬프트 템플릿을 적용해서 데이터셋 포맷으로 만들어 반환하는 함수입니다.


def simple_prompt(example):
    PROMPT = ""
    if example["paragraph"]:
        PROMPT += """
            지문:
            {paragraph}
            """
    PROMPT += """
        질문:
        {question}
        """
    if example["question_plus"]:
        PROMPT += """
            <보기>:
            {question_plus}
            """
    PROMPT += """
        선택지:
        {choices}
        """
    if len(example["choices"]) == 4:
        PROMPT += """
            1, 2, 3, 4 중에 하나를 정답으로 고르세요.

            정답:
            """
    else:
        PROMPT += """
            1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.

            정답:
            """

    choices_string = "\n".join(
        [f"{idx + 1} - {choice}" for idx, choice in enumerate(example["choices"])]
    )
    len_choices = len(example["choices"])
    if example["paragraph"] and example["question_plus"]:
        user_message = PROMPT.format(
            paragraph=example["paragraph"],
            question=example["question"],
            question_plus=example["question_plus"],
            choices=choices_string,
        )
    elif example["question_plus"]:
        user_message = PROMPT.format(
            question=example["question"],
            question_plus=example["question_plus"],
            choices=choices_string,
        )
    elif example["paragraph"]:
        user_message = PROMPT.format(
            paragraph=example["paragraph"],
            question=example["question"],
            choices=choices_string,
        )
    else:
        user_message = PROMPT.format(
            question=example["question"],
            choices=choices_string,
        )

    messages = [
        {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
        {"role": "user", "content": user_message.strip()},
    ]
    if example["answer"]:
        messages.append({"role": "assistant", "content": f"{example['answer']}"})

    chat_message = {
        "id": example["id"],
        "messages": messages,
        "label": example["answer"],
        "len_choices": len_choices,
    }
    return chat_message
