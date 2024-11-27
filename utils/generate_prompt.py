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
    if "answer" in example.keys():
        messages.append({"role": "assistant", "content": f"{example['answer']}"})

    chat_message = {
        "id": example["id"],
        "messages": messages,
        "len_choices": len_choices,
    }
    return chat_message


def cot_prompt(example):
    PROMPT = ""
    if example["paragraph"]:
        PROMPT += "지문:\n{paragraph}\n"
    PROMPT += "질문:\n{question}\n"
    if example["question_plus"]:
        PROMPT += "<보기>:\n{question_plus}\n"
    PROMPT += "선택지:\n{choices}\n"
    if len(example["choices"]) == 4:
        PROMPT += "1, 2, 3, 4 중에 하나를 정답으로 고르세요.\n\n<문제 풀이>:\n"
    else:
        PROMPT += "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n\n<문제 풀이>:\n"

    instruction = """
    문제를 읽고 다음의 구조에 맞추어 적절한 문제 풀이와 정답을 작성하세요.:

    지문 분석:
    - 지문의 핵심 내용을 요약하고, 문제 해결에 필요한 중요한 단서를 명확히 파악합니다.
    - 지문 속 키워드나 개념이 어떤 맥락에서 제시되었는지 서술합니다.

    선택지 검토:
    - 각 선택지를 하나씩 검토하며, 옳고 그름을 판단하는 근거를 명확히 제시합니다.
    - 관련된 지문 내용과 역사적/과학적/문학적 배경지식을 활용하여 논리적으로 설명합니다.

    정답 도출:
    -  선택지 중 가장 적합한 답을 선택하고, 이를 지문 내용과 문제 요구 사항에 근거하여 명확히 설명합니다.

    정답:
    - 정답은 1, 2, 3, 4, 5 중 하나의 숫자로 작성합니다.

    반드시 문제 풀이를 먼저 작성하고, 마지막에 정답: {정답 숫자}의 형태로 정답을 출력하세요.
    """

    choices_string = "\n".join(
        [f"{idx + 1} - {choice}" for idx, choice in enumerate(example["choices"])]
    )

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
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_message.strip()},
    ]

    if "solving" in example.keys():
        assistant_message = f"{example['solving']}\n\n"
        if "answer" in example.keys() and example["answer"]:
            assistant_message += f"정답: {example['answer']}"
        messages.append({"role": "assistant", "content": assistant_message})

    chat_message = {
        "id": example["id"],
        "messages": messages,
        "len_choices": len(example["choices"]),
    }
    return chat_message


def dpo_prompt(example):
    PROMPT = ""
    if example["paragraph"]:
        PROMPT += "지문:\n{paragraph}\n"
    PROMPT += "질문:\n{question}\n"
    if example["question_plus"]:
        PROMPT += "<보기>:\n{question_plus}\n"
    PROMPT += "선택지:\n{choices}\n"
    if len(example["choices"]) == 4:
        PROMPT += "1, 2, 3, 4 중에 하나를 정답으로 고르세요.\n\n<문제 풀이>:\n"
    else:
        PROMPT += "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n\n<문제 풀이>:\n"

    instruction = """
    문제를 읽고 다음의 구조에 맞추어 적절한 문제 풀이와 정답을 작성하세요.:

    지문 분석:
    - 지문의 핵심 내용을 요약하고, 문제 해결에 필요한 중요한 단서를 명확히 파악합니다.
    - 지문 속 키워드나 개념이 어떤 맥락에서 제시되었는지 서술합니다.

    선택지 검토:
    - 각 선택지를 하나씩 검토하며, 옳고 그름을 판단하는 근거를 명확히 제시합니다.
    - 관련된 지문 내용과 역사적/과학적/문학적 배경지식을 활용하여 논리적으로 설명합니다.

    정답 도출:
    -  선택지 중 가장 적합한 답을 선택하고, 이를 지문 내용과 문제 요구 사항에 근거하여 명확히 설명합니다.

    정답:
    - 정답은 1, 2, 3, 4, 5 중 하나의 숫자로 작성합니다.

    반드시 문제 풀이를 먼저 작성하고, 마지막에 정답: {정답 숫자}의 형태로 정답을 출력하세요.
    """

    choices_string = "\n".join(
        [f"{idx + 1} - {choice}" for idx, choice in enumerate(example["choices"])]
    )

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

    prompt = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_message.strip()},
    ]

    assistant_response = """
    {solving}

    정답: {answer}
    """
    chosen = [
        {"role": "assistant", "content": assistant_response.format(solving=example["chosen"], answer=example["answer"])}
    ]

    rejected = [
        {"role": "assistant", "content": assistant_response.format(solving=example["rejected"], answer=example["answer"])}
    ]
    
    chat_message = {
        "id": example["id"],
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "len_choices": len(example["choices"]),
    }
    return chat_message
