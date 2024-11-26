from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# 모델 이름 설정
model_name = "Qwen/Qwen2.5-32B-Instruct"

# 모델 및 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 프롬프트 설정
def generate_prompt(input_text, question):
    return f"""
    당신은 주어진 데이터가 속하는 분야를 구분할 수 있는 어시스턴트로, 
    각 데이터가 사회 분야에 속하는지 생각하고 RAG에 이용하기 위한 키워드를 추출해야 합니다.

    질문: {question}

    지문: {input_text}

    데이터를 보고 이 문제가 사회 분야에 속할지 근거를 찾아 판단하고,
    사회 분야에 속한다면 1을 반환하고 그렇지 않다면 0을 반환해야 합니다.
    만약 사회 분야에 속한다면, 데이터로부터 주요 키워드를 5개 추출해 반환해야 합니다. 사회 분야에 속하지 않는다면 키워드는 None입니다.

    응답 형식:
    키워드: 키워드1, 키워드2, 키워드3, 키워드4, 키워드5
    is_social: 0 또는 1
    """

# 응답 생성 함수
def generate_response(input_text, question):
    # 프롬프트 생성
    prompt = generate_prompt(input_text, question)
    
    # 시스템 및 사용자 메시지 설정
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # 텍스트 생성
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 모델 입력 처리
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 텍스트 생성
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    # 생성된 토큰을 디코딩하여 텍스트로 변환
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # 텍스트 디코딩
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 예시 입력
input_text = "사람들이 지속적으로 책을 읽는 이유 중 하나는 즐거움이다. 독서의 즐거움에는 여러 가지가 있겠지만 그 중심에는 ‘소통의 즐거움’이 있다."
question = "윗글의 내용과 일치하지 않는 것은?"

# 응답 생성
response = generate_response(input_text, question)
print("응답:", response)

# 응답에서 키워드와 사회 여부 추출 (정규식 사용)
def extract_keywords_and_is_social(response):
    # 키워드 추출
    keyword_match = re.search(r"키워드:\s*([^,]+(?:, [^,]+)*)", response)
    is_social_match = re.search(r'is_social:\s*(\d)', response)
    
    if keyword_match and is_social_match:
        extracted_keywords = keyword_match.group(1).strip()
        is_social_value = int(is_social_match.group(1))
    else:
        extracted_keywords = None
        is_social_value = None
    
    return extracted_keywords, is_social_value

# 키워드와 사회 여부 추출
keywords, is_social = extract_keywords_and_is_social(response)
print("추출된 키워드:", keywords)
print("사회 여부:", is_social)
