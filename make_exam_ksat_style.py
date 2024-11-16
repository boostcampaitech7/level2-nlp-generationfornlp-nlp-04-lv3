from swarm import Swarm, Agent

client = Swarm()


def transfer_to_factual():
    """transfer to factual_comprehension_agent for making Factual Comprehension Question"""
    return factual_comprehension_agent


def transfer_to_inferential():
    """Transfer to inferential_comprehension_agent for making Inferential Comprehension Question."""
    return inferential_comprehension_agent


def transfer_to_critical():
    """Transfer to critical_comprehension_agent for making Critical Comprehension Question."""
    return critical_comprehension_agent


def transfer_to_creative():
    """Transfer to creative_application_agent for making Creative Application Question."""
    return creative_application_agent


supervisor = Agent(
    name="Korean CSAT Problem Supervisor",
    instructions=(
        "Receive a paragraph and distribute it to the appropriate assistant agent "
        "based on the cognitive domain of the problem to be generated."
    ),
    functions=[
        transfer_to_creative,
        transfer_to_critical,
        transfer_to_factual,
        transfer_to_inferential,
    ],
    model="qwen2.5:32b",
)

# 사실적 이해 문제 생성 에이전트 정의
factual_comprehension_agent = Agent(
    name="Factual Comprehension Question Assistant",
    instructions=(
        "Answer in korean. Analyze the given paragraph and generate a question that evaluates "
        "the ability to accurately understand the information, structure, and relationships presented in the text. "
        "Strictly adhere to output formatting"
        "Output Format:"
        "<Question>: question"
        "1. <choices_1>"
        "2. <choices_2>"
        "3. <choices_3>"
        "4. <choices_4>"
        "5. <choices_5>"
        "<answer>: number"
    ),
    model="qwen2.5:32b",
)

# 추론적 이해 문제 생성 에이전트 정의
inferential_comprehension_agent = Agent(
    name="Inferential Comprehension Question Assistant",
    instructions=(
        "Answer in korean. Analyze the given paragraph and generate a question that evaluates "
        "the ability to logically infer information not explicitly stated in the text, including implicit meanings and contextual implications. "
        "Strictly adhere to output formatting"
        "Output Format:"
        "<Question>: question"
        "1. <choices_1>"
        "2. <choices_2>"
        "3. <choices_3>"
        "4. <choices_4>"
        "5. <choices_5>"
        "<answer>: number"
    ),
    model="qwen2.5:32b",
)

# 비판적 이해 문제 생성 에이전트 정의
critical_comprehension_agent = Agent(
    name="Critical Comprehension Question Assistant",
    instructions=(
        "Answer in korean. Analyze the given paragraph and generate a question that evaluates "
        "the ability to critically interpret the content and form of the text, assessing its validity, appropriateness, and value. "
        "Strictly adhere to output formatting"
        "Output Format:"
        "<Question>: question"
        "1. <choices_1>"
        "2. <choices_2>"
        "3. <choices_3>"
        "4. <choices_4>"
        "5. <choices_5>"
        "<answer>: number"
    ),
    model="qwen2.5:32b",
)


# 적용･창의 문제 생성 에이전트 정의
creative_application_agent = Agent(
    name="Creative Application Question Assistant",
    instructions=(
        "Answer in korean."
        "Create questions in the style of Korean SAT questions."
        "Analyze the given paragraph and generate a question that evaluates the ability to creatively apply or utilize the concepts and principles of the text in a new context, including content generation and expression."
        "Use a different field and different words than the given paragraph."
        "There must be at least one compelling wrong answer."
        "Create a clear rationale for your answer before you print the correct answer. The answer shouldn't be forced."
        "Strictly adhere to output formatting"
        "Output Format:"
        "<Question>: question"
        "1. <choices_1>"
        "2. <choices_2>"
        "3. <choices_3>"
        "4. <choices_4>"
        "5. <choices_5>"
        "<answer>: rationale, number"
        "<compelling wrong answer>: "
    ),
    model="qwen2.5:32b",
)


def route_paragraph_and_generate(paragraph, domain):
    """
    Routes the paragraph to the appropriate agent based on the domain
    and generates a question.
    """
    # Routing logic based on the domain
    if domain == "factual":
        agent = transfer_to_factual()
    elif domain == "inferential":
        agent = transfer_to_inferential()
    elif domain == "critical":
        agent = transfer_to_critical()
    elif domain == "creative":
        agent = transfer_to_creative()
    else:
        raise ValueError(
            "Invalid domain specified. Choose from 'factual', 'inferential', 'critical', 'creative'."
        )

    # Run the selected agent
    response = client.run(
        agent=agent,
        messages=[
            {
                "role": "user",
                "content": f"Analyze the following paragraph and generate a question:\n\n{paragraph}",
            }
        ],
    )

    # Extract and return the generated question and answer choices
    return response.messages[-1]["content"]


# Example Usage
paragraph = (
    "대법원이 전국교직원노동조합(전교조)에 대한 고용노동부의 법외노조 통보 효력을 확정판결 전까지 잠정 중단하라고 한 서울고등법원 결정을 뒤집었다."
    "대법원 1부(주심 고영한 대법관)는 고용부가 전교조 법외노조 통보 처분의 효력정지 결정에 반발해 제기한 재항고소송에서 전교조의 손을 들어준 원심을 깨고 사건을 서울고법으로 돌려보냈다고 3일 발표했다."
    "고용부는 2013년 10월 전교조가 해직 교원 9명을 노조원으로 포함하고 있다는 이유로 법외노조라고 통보했다."
    "이에 전교조는 법외노조 통보를 취소하라며 소송을 냈다. 1심에서는 전교조가 패소했지만 항소심에서 위헌법률심판제청 신청과 효력정지 신청이 받아들여졌다."
    "이에 따라 전교조는 합법 노조 지위를 유지한 상태에서 헌재 결정을 기다렸다."
    "그러나 헌재는 지난달 교원노조법 2조에 대해 합헌 결정을 했다.재판부는 “원심이 효력정지를 인정한 건 ‘해당 조항이 헌법에 위반된다고 의심할 만한 상당한 이유가 있어 위헌법률심판 제청을 했다’는 점을 전제로 한 것”이라며 "
    "헌재가 합헌 결정을 했으므로 원심의 판단에는 집행정지에 관한 법리를 오해한 위법이 있다”고 지적했다."
)

paragraph_1 = (
    "극심한 취업난을 겪고 있는 인문·사회 계열의 문과 대학생들이 약학대학입문자격시험(PEET)에 몰리고 있다."
    "PEET는 2009년 약학대학 학제가 4년제에서 6년제로 개편되면서 도입된 약학대학 3학년 편입학 시험이다."
    "2012년 서울대 경영대학을 졸업한 최모씨(29)는 최근 PEET 학원을 돌며 ‘약대 진학’ 상담을 받고 있다."
    "2011년 하반기부터 20여개 기업에 지원했지만 번번이 고배를 마셨다."
    "최씨는 “차라리 좀 더 공부해서 전문직이 되는 것이 어떠냐는 부모님 조언에 따라 PEET를 준비하기 시작했다”며 “의사가 될 수 있는 의학전문대학원 시험보다는 약대 시험이 조금은 수월하다고 판단했다”고 말했다."
    "PEET 준비 학생들이 모이는 ‘약대가자’ 등 인터넷 카페에도 시험을 준비하는 문과생들의 글이 적지 않게 올라온다."
    "‘인문대 재학생인데 복수전공을 생명공학으로 돌려서 약대를 지원하면 가망이 있을까요?’라는 글처럼 약대 진학을 위한 스펙과 필수 수업 등에 대한 문의가 많다."
    "약대에 편입하려는 문과 대학생들이 늘면서 PEET 응시자 수는 사상 최고치를 기록했다."
    "한국약학교육협의회에 따르면 올해 시험 응시자는 1만4706명으로 지난해(1만4330명)보다 증가했다."
    "이공계를 제외한 비전공자 응시자 수도 2012년 이후 3년 연속 2000명을 넘어섰다."
    "이런 분위기를 타고 PEET 학원들은 ‘문과생 모시기’에 적극 나서고 있다."
    "지난 4일 신촌의 한 PEET 준비학원은 ‘문과생을 위한 약대 합격 설명회’를 열었다."
    "이날 설명회 홍보포스터 등에는 ‘언제까지 미래를 보장받지 못하는 공부에 올인하실건가요?’라는 파격적인 문구가 씌어 있었다."
    "설명회엔 100여명의 학생이 몰렸다.강남의 한 PEET 준비학원은 비전공자 과정 수업을 별도로 진행하고 있다."
    "PEET가 생물학, 화학, 유기화학, 물리학 등으로 구성되다 보니 바로 수업을 따라가기 어려운 문과생들을 위한 과정을 만든 것이다."
    "이찬 서울대 산업인력개발학과 교수는 “취업난이 심해지면서 상대적으로 안정적 직업인 약사가 되려는 학생들이 편입에 적극적인 것 같다”며 “하지만 전문직도 취업에 어려움을 겪는 경우가 많기 때문에 심사숙고해 결정해야 한다”고 말했다."
)

# Call the function to generate a question
domain = "inferential"  # Change to  "factual", "inferential", "critical", or "creative" as needed
question_output = route_paragraph_and_generate(paragraph_1, domain)
print("Generated Question and Answer Choices:")
print(question_output)
