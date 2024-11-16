import csv
import ast
import re
from swarm import Swarm, Agent
from tqdm import tqdm

# Initialize the Swarm client and agents (reuse collaborative_solver function from before)
client = Swarm()

spec_generator = Agent(
    name="Specification Generator",
    instructions=(
        "Based on the given context, question, and choices, generate a detailed functional specification "
        "that describes how to solve the problem. The specification should include logical steps, necessary checks, "
        "and expected outcomes for solving the question."
        "Let's think about step by step"
    ),
    model="qwen2.5:32b",
)

problem_solver = Agent(
    name="Problem Solver",
    instructions=(
        "Using the provided functional specification, solve the given problem and return a clear, formatted answer. "
        "If the solution involves selecting a choice, provide the number corresponding to the choice (1, 2, 3, 4, 5)."
        "Only print number"
    ),
    model="qwen2.5:32b",
)

verifier_agent = Agent(
    name="Verifier",
    instructions=(
        "Verify the correctness of the provided solution against the functional specification and problem details. "
        "If the solution is correct, confirm it and only print number. If incorrect, explain why and suggest adjustments to the functional "
        "specification for re-solving."
        "Let's think about step by step"
    ),
    model="qwen2.5:32b",
)


def collaborative_solver(context, question, choices):
    """
    Executes a multi-step solving process involving functional specification generation,
    problem solving, and verification.
    """
    spec_response = client.run(
        agent=spec_generator,
        messages=[
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}\n\nChoices: {choices}",
            }
        ],
    )
    functional_spec = spec_response.messages[-1]["content"]
    solution_response = client.run(
        agent=problem_solver,
        messages=[
            {
                "role": "user",
                "content": f"Functional Specification: {functional_spec}\n\nQuestion: {question}\n\nChoices: {choices}",
            }
        ],
    )
    solution = solution_response.messages[-1]["content"]
    verification_response = client.run(
        agent=verifier_agent,
        messages=[
            {
                "role": "user",
                "content": f"Functional Specification: {functional_spec}\n\nProposed Solution: {solution}\n\nQuestion: {question}\n\nChoices: {choices}",
            }
        ],
    )
    verification_result = verification_response.messages[-1]["content"]
    if "incorrect" in verification_result.lower():
        return collaborative_solver(
            context, question, choices
        )  # Re-solve iteratively if necessary
    return solution


# File paths
input_file = "/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/test.csv"
output_file = "output_results.csv"

# Read and process the file
results = []
# max_process_count = 5  # 처리할 최대 데이터 수
# current_count = 0  # 처리한 데이터 개수 카운터
with open(input_file, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)  # 전체 데이터를 리스트로 읽어오기
    total_rows = len(rows)  # 총 데이터 개수

    for row in tqdm(rows, total=total_rows, desc="Processing rows"):
        # if current_count >= max_process_count:  # 5개까지만 처리
        #     break
        id_ = row["id"]
        paragraph = row["paragraph"]
        try:
            problems = ast.literal_eval(row["problems"])  # Parse safely
        except Exception as e:
            print(f"Error parsing 'problems' field for ID {id_}: {e}")
            problems = {"question": "", "choices": []}
        question = problems["question"]
        choices = problems["choices"]
        question_plus = row.get("question_plus", "")

        # Combine contexts if `question_plus` exists
        context = f"{paragraph}\n\n{question_plus}" if question_plus else paragraph

        # Solve the problem using the collaborative_solver
        print(f"Processing ID: {id_}")  # For debug progress
        try:
            final_answer = collaborative_solver(context, question, choices)

            # 번호만 추출 (정규식 사용)
            number_match = re.search(r"\b\d+\b", final_answer)
            if number_match:
                final_answer = int(number_match.group())  # 번호를 int로 변환
            else:
                final_answer = None  # 번호를 찾지 못하면 None으로 처리
        except Exception as e:
            final_answer = None

        # Collect results
        results.append({"id": id_, "answer": final_answer})
        # current_count += 1  # 데이터 처리 카운터 증가

# Save results to a new CSV file
with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["id", "answer"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {output_file}")
