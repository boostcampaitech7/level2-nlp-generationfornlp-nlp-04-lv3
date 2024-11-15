from swarm import Swarm, Agent

client = Swarm()

analyze_question = Agent(
    name="Question analysis assistant",
    instructions="Below is the given article and question. In 500 words or less, write an understandable explanation that includes a clear rationale for solving this problem.",
    model="qwen2.5:32b",
)

student_agent = Agent(
    name="Student Assistant",
    instructions="Based on the explanation provided, pick the number (1, 2, 3, 4, 5) that best describes your answer. Strictly adhere to the output format.",
    model="qwen2.5:32b",
)


# 함수 생성(function)
# Define a function for collaboration between agents
def output_number(context, question, choices):
    """
    Use the analyze_question agent to generate an explanation, and pass it to the student_agent to select an answer.
    """
    # Step 1: Analyze the question
    explanation_response = client.run(
        agent=analyze_question,
        messages=[
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}\n\nChoices: {choices}",
            }
        ],
    )

    explanation = explanation_response.messages[-1]["content"]
    print(explanation)
    # Step 2: Use explanation to pick the number
    answer_response = client.run(
        agent=student_agent,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Explanation: {explanation}\n\n"
                    f"Question: {question}\n\n"
                    f"Choices: {choices}"
                ),
            }
        ],
    )

    # Return the final numerical answer
    return answer_response.messages[-1]["content"]


context = (
    "수도요금과 생산 원가의 격차가 최근 10년래 가장 큰 것으로 나타났다. 이에 따라 수도 요금 인상 필요성이 높아지는 것으로 조사됐다. "
    "환경부가 10일 발표한 ‘2011년 상하수도 통계’에 따르면 2011년 수도요금은 1㎥(1000ℓ)당 619.3원으로 전년 대비 9.1원 올랐으나 "
    "생산 원가는 36.2원 오른 813.4원을 기록, 원가 대비 요금 비율(요금현실화율)이 78.5%에서 76.1%로 떨어졌다. "
    "생산 원가 대비 수도요금 비율은 2003년 89.3%를 정점으로 계속 떨어지고 있다. 특히 2011년 요금현실화율 76.1%는 최근 10년래 가장 낮은 비율이다. "
    "하수도의 경우도 마찬가지였다. 하수도 전국 평균 요금은 t당 289.4원으로 평균 하수처리원가(807.1원)의 35.8%(요금현실화율)에 불과했다. "
    "비용 부담으로 인해 수도를 담당하는 지자체의 재정도 악화되고 있다. 서울시의 경우 수도 관련 부채가 2011년 현재 3227억원에 달해 지자체 중 가장 많았다. "
    "이어 전남(1152억원), 경기(787억원), 대구(771억원) 등의 순이었다. 반면 인천과 부산은 부채가 하나도 없었다. "
    "비용이 증가하는 이유는 수도 관리에 필요한 전기요금이 상승하는 반면 요금 인상은 제대로 이뤄지지 않고 있기 때문이다. "
    "현재 상하수도 관리는 각 지자체가 하고 있으며 요금은 시의회의 승인을 받아 지자체 장이 결정하는 구조로 돼 있다."
)

question = "2011년 수도요금의 요금현실화율은 얼마였는가?"
choices = ["76.1%", "78.5%", "89.3%", "35.8%", "100%"]
# Context
context1 = (
    "사람들이 지속적으로 책을 읽는 이유 중 하나는 즐거움이다. 독서의 즐거움에는 여러 가지가 있겠지만 그 중심에는 ‘소통의 즐거움’이 있다. "
    "독자는 독서를 통해 책과 소통하는 즐거움을 경험한다. 독서는 필자와 간접적으로 대화하는 소통 행위이다. 독자는 자신이 속한 사회나 시대의 "
    "영향 아래 필자가 속해 있거나 드러내고자 하는 사회나 시대를 경험한다. 직접 경험하지 못했던 다양한 삶을 필자를 매개로 만나고 이해하면서 "
    "독자는 더 넓은 시야로 세계를 바라볼 수 있다. 이때 같은 책을 읽은 독자라도 독자의 배경 지식이나 관점 등의 독자 요인, 읽기 환경이나 "
    "과제 등의 상황 요인이 다르므로, 필자가 보여 주는 세계를 그대로 수용하지 않고 저마다 소통 과정에서 다른 의미를 구성할 수 있다. 이러한 "
    "소통은 독자가 책의 내용에 대해 질문하고 답을 찾아내는 과정에서 가능해진다. 독자는 책에서 답을 찾는 질문, 독자 자신에게서 답을 찾는 "
    "질문 등을 제기할 수 있다. 전자의 경우 책에 명시된 내용에서 답을 발견할 수 있고, 책의 내용들을 관계 지으며 답에 해당하는 내용을 스스로 "
    "구성할 수도 있다. 또한 후자의 경우 책에는 없는 독자의 경험에서 답을 찾을 수 있다. 이런 질문들을 풍부히 생성하고 주체적으로 답을 찾을 때 "
    "소통의 즐거움은 더 커진다. 한편 독자는 ㉠다른 독자와 소통하는 즐거움을 경험할 수도 있다. 책과의 소통을 통해 개인적으로 형성한 의미를 독서 "
    "모임이나 독서 동아리 등에서 다른 독자들과 나누는 일이 이에 해당한다. 비슷한 해석에 서로 공감하며 기존 인식을 강화하거나 관점의 차이를 "
    "확인하고 기존 인식을 조정하는 과정에서, 독자는 자신의 인식을 심화･확장할 수 있다. 최근 소통 공간이 온라인으로 확대되면서 독서를 통해 "
    "다른 독자들과 소통하며 즐거움을 누리는 양상이 더 다양해지고 있다. 자신의 독서 경험을 담은 글이나 동영상을 생산･공유함으로써, 책을 읽지 "
    "않은 타인이 책과 소통하도록 돕는 것도 책을 통한 소통의 즐거움을 나누는 일이다."
)

# Question
question1 = "윗글의 내용과 일치하지 않는 것은?"
question2 = "윗글을 읽고 ㉠에 대해 보인 반응으로 적절하지 않은  것은?"
# Choices
choices1 = [
    "같은 책을 읽은 독자라도 서로 다른 의미를 구성할 수 있다.",
    "다른 독자와의 소통은 독자가 인식의 폭을 확장하도록 돕는다.",
    "독자는 직접 경험해 보지 못했던 다양한 삶을 책의 필자를 매개로 접할 수 있다.",
    "독자의 배경지식, 관점, 읽기 환경, 과제는 독자의 의미 구성에 영향을 주는 독자 요인이다.",
    "독자는 책을 읽을 때 자신이 속한 사회나 시대의 영향을 받으며 필자와 간접적으로 대화한다.",
]
choices2 = [
    "스스로 독서 계획을 세우고 자신에게 필요한 책을 찾아 개인적 으로 읽는 과정에서 경험할 수 있겠군.",
    "독서 모임에서 서로 다른 관점을 확인하고 자신의 관점을  조정하는 과정에서 경험할 수 있겠군.",
    "개인적으로 형성한 의미를,  독서 동아리를 통해 심화하는  과정에서 경험할 수 있겠군.",
    "자신의 독서 경험을 담은 콘텐츠를 생산하고 공유하는 과정 에서 경험할 수 있겠군.",
    "오프라인뿐 아니라 온라인 공간에서 해석을 나누는 과정에서도  경험할 수 있겠군.",
]

# Call the function and print the result
answer = output_number(context=context1, question=question1, choices=choices1)
print("Final Answer:", answer)
