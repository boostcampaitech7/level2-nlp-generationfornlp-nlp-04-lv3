import csv
import ast
import re
from swarm import Swarm, Agent
from tqdm import tqdm
from langchain_community.tools import TavilySearchResults

model_name = "phi3:14b-medium-128k-instruct-fp16"
data_name = "train"
prompt_name = "plain_text_prompt"
reversed = True
# Initialize the Swarm client and agents (reuse collaborative_solver function from before)
client = Swarm()


def search_on_wiki(query: str):
    """Search 'query' on Wikipedia and return the results"""
    import wikipediaapi

    print("====================================")
    print("query")
    print(query)
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="MyCustomUserAgent/1.0 (myemail@example.com)", language="ko"
    )  # 'en'은 영어, 필요하면 'ko'로 변경
    page = wiki_wiki.page(query)

    if page.exists():
        result = f"Title: {page.title}\n\nSummary: {page.summary[:1000]}..."  # 요약 500자 제한
    else:
        result = "No results found on Wikipedia for the given query."

    print("====================================")
    print("Wiki Result")
    print(result)
    return result


def search_on_web(query: str):
    """Search "query' on the web google) and return the results"""
    search_tool = TavilySearchResults(
        max_results=3,
        extra_params=dict(
            api_key="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",
            include_answer=True,
            search_depth="advanced",
            include_domains=["google.com", "wikipedia.org"],
        ),
    )
    result = search_tool.invoke(query)
    contents = [item["content"] for item in result]
    print("====================================")
    print("Web Result")
    print(contents)
    return contents


def transfer_to_verifier():
    """Transfers control to the verifier agent for verify answer."""
    return verifier_agent


def transfer_to_solver():
    """Transfers control to the problem solver agent for solve problem."""
    return problem_solver


spec_generator_prompt = """
Given the context, question, and choices, sequence the process you would take to solve the problem.
First, summarize the context in 1000 characters long.
Second, Your process should include the logical steps to solve the problem, the checks you'll need to make, and the steps you'll need to take. Think step by step.
Make sure your output is no more than 1500 characters long. And strictly adhere to the output format.
"""
"""주어진 Context, question, choices를 바탕으로 문제를 해결하기 위해 밟아야할 프로세스를 순서대로 작성합니다.
프로세스에는 문제 해결을 위한 논리적 단계, 필요한 확인 사항 포함되어야 합니다. 단계별로 생각해 봅시다.
500자 이하로 작성하고 출력형식을 엄격히 지키세요."""

plain_text_prompt = """
Ignore Previous Chat.
Answer the given question but provide only the number of the correct choice as the output. Do not include any explanation, description, or additional text. Strictly follow this format:

Choices: 1, 2, 3, 4, 5
Output format: A single number representing the correct choice
Restriction:
Important:
You must output only the number. Any additional explanation, description, or deviation from this rule is not allowed.
Example:
Input:
Context: {context}
Question: {question}
Choices: {numbered_choices}
Output: 2
"""

XML_prompt_base = """
Ignore Previous Chat.
<prompt>
    <instructions>
        Answer the given question but provide only the number of the correct choice as the output.
        Do not include any explanation, description, or additional text. Strictly follow this format:
    </instructions>
    <format>
        <choices>1, 2, 3, 4, 5</choices>
        <output_format>A single number representing the correct choice</output_format>
    </format>
    <restriction>
        <important>
            You must output <only>the number</only>. Any additional explanation, description, or deviation from this rule is not allowed.
        </important>
    </restriction>
    <example>
        <input>
            <context>\{context\}</context>
            <question>\{question\}</question>
            <choices>\{numbered_choices\}</choices>
        </input>
        <output>2</output>
    </example>
    <methodology>
        Follow this methodology for all problems provided to you.
        Return strictly and only the number of the correct choice as the output.
    </methodology>
</prompt>
"""
XML_prompt_CoT = """
Ignore Previous Chat.
<prompt>
    <instructions>
        Task: choose the correct answer
        Approach: Let's solve this step by step
        Instructions:
        1. Break down the problem into sub-components
        2. Evaluate each step
        3. Consider multiple solution paths
        4. Validate intermediate results
        5. Synthesize final answer
        Please show your reasoning for each step.
    </instructions>
    <format>
        <choices>1, 2, 3, 4, 5</choices>
        <output_format>A single number representing the correct choice</output_format>
    </format>
    <restriction>
        <important>
            You must output <only>the number</only> at last after your step-by-step reasoning.
        </important>
    </restriction>
    <example>
        <input>
            <context>{context}</context>
            <question>{question}</question>
            <choices>{numbered_choices}</choices>
        </input>
        <output>
            [Step-by-step reasoning here]
            2
        </output>
    </example>
    <methodology>
        Follow this methodology for all problems provided to you.
        Show your step-by-step reasoning, then return strictly and only the number of the correct choice as the final output.
    </methodology>
</prompt>
"""
XML_problem_solver_prompt = """
Ignore Previous Chat.
<prompt>
    <instructions>
        Answer the given problem step by step internally. If you lack information, use the search_on_web function.
        Provide only the number of the correct choice as the output.
        Do not include any explanation, description, or additional text. Strictly follow this format:
    </instructions>
    <format>
        <choices>1, 2, 3, 4, 5</choices>
        <output_format>A single number representing the correct choice</output_format>
    </format>
    <restriction>
        <important>
            You must output <only>the number</only>. Any additional explanation, description, or deviation from this rule is not allowed.
        </important>
    </restriction>
    <example>
        <input>
            <context>{context}</context>
            <question>{question}</question>
            <choices>{numbered_choices}</choices>
        </input>
        <output>2</output>
    </example>
    <methodology>
        1. Analyze the given context and question.
        2. If additional information is needed, use search_on_web(query) to find relevant details.
        3. Process the information and determine the correct answer.
        4. Return strictly and only the number of the correct choice as the output.
    </methodology>
    <function>
        <name>search_on_web</name>
        <description>Use this function to search for additional information if needed. Example: search_on_web("query")</description>
    </function>
</prompt>
"""
Top_down_XML_problem_solver_prompt = """
<prompt>
    <instructions>
        Identify the question and choices, read the context, and print the number of incorrect choices.
        Do not include any explanation, description, or additional text. Strictly follow this format:
    </instructions>
    <format>
        <choices>1, 2, 3, 4, 5</choices>
        <output_format>A single number representing the correct choice</output_format>
    </format>
    <restriction>
        <important>
            You must output <only>the number</only>. Any additional explanation, description, or deviation from this rule is not allowed.
        </important>
    </restriction>
    <example>
        <input>
            <question>{question}</question>
            <choices>{numbered_choices}</choices>
            <context>{context}</context>
        </input>
        <output>2</output>
    </example>
    <methodology>
        1. Analyze the given question and choices.
        2. Read the context and determine the correct answer.
        3. Return strictly and only the number of the correct choice as the output.
    </methodology>
</prompt>
"""
"""당신은 번호만 출력하는 문제 풀이 기계입니다. 정답이외의 다른 설명은 포함하지마세요. 주어진 프로세스를 사용하여 주어진 문제를 단계별로 풀고 답을 반환합니다.
Choices는 리스트의 순서대로 1, 2, 3, 4, 5를 의미합니다.
출력은 선택 번호(1, 2, 3, 4, 5)만 포함해야 합니다.
출력 형식을 엄격하게 준수하세요."""

plain_text_problem_solver_prompt = """
You are a problem-solving machine that only outputs numbers. Don't include any explanation other than the answer. Solve the given problem step by step using the given process and return the answer.
Choices means 1, 2, 3, 4, 5 in the order of the list.
The output should contain only the choice numbers 1~5.
## 문제 예시
거짓 문장을 선택하십시오
① 진실 문장
✔ 거짓 문장
③ 진실 문장
④ 진실 문장
⑤ 진실 문장
<output>2</output>
진실 문장을 선택하십시오
① 거짓 문장
② 거짓 문장
✔ 진실 문장
④ 거짓 문장
⑤ 거짓 문장
<output>3</output>
"""

verifier_agent_prompt = """
Verify that the provided predicted solution is correct based on contest, question.
I need you to check it out and print out the correct answer.
Example:
input:
Predicted Solution: {functional_spec}

Context: {context}

Question: {question}

Choices: {choices}
output: 2
---
Just print out the right single number. Don't include any explanation.
"""
"""제공된 솔루션이 프로세스와 문제 세부 정보에 근거해서 맞는지 확인합니다.
솔루션이 정확하면 번호만 출력합니다. 정확하지 않은 경우 incorrect를 출력하고 이유를 설명하고 프로세스에 대한 조정을 제안합니다.
500자 이내로 작성하고 출력 형식을 엄격히 지키세요.
단계별로 생각해 봅시다."""

final_solver_prompt = """
You need to solve the problem again based on your previous solution and the reason why it is wrong.
The output format should contain only the choice numbers (1, 2, 3, 4, 5).
Strictly adhere to the output format.
"""
spec_generator = Agent(
    name="Specification Generator",
    instructions=spec_generator_prompt.strip(),
    model="qwen2.5:32b",
)

problem_solver = Agent(
    name="Problem Solver",
    instructions=globals()[prompt_name].strip(),
    model=model_name,
    # functions=[transfer_to_verifier, search_on_web, search_on_wiki],
)
verifier_agent = Agent(
    name="Verifier",
    instructions=verifier_agent_prompt.strip(),
    model=model_name,
)
final_solver_agent = Agent(
    name="Final Solver",
    instructions=final_solver_prompt.strip(),
    model="qwen2.5:14b",
)


def final_solver(context, question, choices, solution, reason):
    final_solve = client.run(
        agent=final_solver_agent,
        messages=[
            {
                "role": "user",
                "content": f"Previous Solution: {solution}\n\nWhy it's wrong: {reason}\n\nContext: {context}\n\nQuestion: {question}\n\nChoices: {choices}",
            }
        ],
    )
    solution = final_solve.messages[-1]["content"]
    print("====================================")
    print("Final Solution")
    print(solution)
    return solution


def collaborative_solver(context, question, choices):
    """
    Executes a multi-step solving process involving functional specification generation,
    problem solving, and verification.
    """
    numbered_choices = [f"{i + 1}. {choice}" for i, choice in enumerate(choices)]
    spec_response = client.run(
        agent=problem_solver,
        messages=[
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}\n\nChoices: {numbered_choices}",
            }
        ],
    )
    functional_spec = spec_response.messages[-1]["content"]
    print("====================================")
    print("functional_spec")
    print(functional_spec)

    return functional_spec


# File paths
input_file = (
    f"/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/{data_name}.csv"
)
if data_name == "test":
    output_file = f"./output/test/{model_name}_{data_name}_reversed_{str(reversed)}_{prompt_name}_output_results.csv"
else:
    output_file = f"./output/{model_name}_{data_name}_reversed_{str(reversed)}_{prompt_name}_output_results.csv"

# Read and process the file
results = []
max_process_count = 5  # Process the maximum number of data (optional)
current_count = 0  # Count the number of processed data (optional)
with open(input_file, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)  # 전체 데이터를 리스트로 읽어오기
    if data_name == "train":
        total_rows = len(rows) // 10  # 총 데이터 개수
    else:
        total_rows = len(rows)

    for row in tqdm(rows[:total_rows], total=total_rows, desc="Processing rows"):
        # if current_count >= max_process_count:  # 5개까지만 처리 (optional)
        #     break
        id_ = row["id"]
        paragraph = row["paragraph"]
        try:
            problems = ast.literal_eval(row["problems"])  # Parse safely
        except Exception as e:
            print(f"Error parsing 'problems' field for ID {id_}: {e}")
            problems = {"question": "", "choices": [], "answer": None}
        question = problems.get("question", "")
        choices = problems.get("choices", [])
        correct_answer = problems.get("answer", None)  # Extract correct answer
        question_plus = row.get("question_plus", "")

        # Combine contexts if `question_plus` exists
        context = (
            f"{paragraph}\n\nPassage:{question_plus}" if question_plus else paragraph
        )

        # Solve the problem using the collaborative_solver
        print(f"Processing ID: {id_}")  # For debug progress
        try:
            final_answer = collaborative_solver(context, question, choices)
            if reversed:
                final_answer = final_answer[::-1]
            # 번호만 추출 (정규식 사용)
            number_match = re.search(r"[1-5]", final_answer)
            print("====================================")
            print("number_match")
            print(number_match)
            if number_match:
                number_match = (
                    number_match.group()[::-1] if reversed else number_match.group()
                )
                final_answer = int(number_match)  # 번호를 int로 변환
            else:
                final_answer = 1  # 번호를 찾지 못하면 None으로 처리
        except Exception as e:
            print(f"Error processing ID {id_}: {e}")
            final_answer = 1
        print("====================================")
        print("final_answer")
        print(final_answer)

        # Collect results with correct answer
        if data_name == "test":
            results.append(
                {
                    "id": id_,
                    "answer": final_answer,
                }
            )
        else:
            results.append(
                {
                    "id": id_,
                    "predicted_answer": final_answer,
                    "correct_answer": correct_answer,
                }
            )
        # current_count += 1  # 데이터 처리 카운터 증가 (optional)

# Save results to a new CSV file
with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    if data_name == "test":
        fieldnames = ["id", "answer"]
    else:
        fieldnames = ["id", "predicted_answer", "correct_answer"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {output_file}")

# Calculate and print the score
total = len(results)
correct = 0
incorrect = 0
if data_name != "test":
    for result in results:
        if (
            result["predicted_answer"] is not None
            and result["predicted_answer"] == result["correct_answer"]
        ):
            correct += 1
        else:
            incorrect += 1

    accuracy = (correct / total) * 100 if total > 0 else 0

    print("====================================")
    print("Scoring Summary")
    print(f"Total Problems: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Incorrect Answers: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")
