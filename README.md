# 수능형 문제 풀이 모델 생성

## 1 프로젝트 개요

**1.1 개요**

본 대회는 국어와 사회 과목을 포함한 수능형 문제 풀이 AI 모델을 구축하는 것을 목표로 한다. 제공된 데이터셋을 활용하여 주어진 지문(paragraph)과 문제(problems)에 기반한 질문(question), 선택지(choices), 정답(answer)을 정확히 예측하는 모델을 개발해야 한다. 

대회에서는 수능(전체 영역), 평가원 모의고사, 수능 관련 문제집(EBS/사설 교재 등)의 데이터 사용이 금지되며, 저작권 문제가 없는 외부 데이터만 활용할 수 있다. 제공된 학습 데이터와 허용된 외부 데이터를 바탕으로 데이터 증강이 가능하며, 유료 API를 사용할 수 있으나 테스트셋 또는 수능 관련 데이터를 기반으로 데이터를 생성하거나 학습에 사용하는 행위는 금지된다.

Ko-MMLU, Multilingual MMLU, KLUE-MRC, 수능 데이터로 학습된 사전학습 가중치는 사용할 수 없으며, 사용 가능한 가중치는 모두 공개적으로 접근 가능해야 한다. 테스트셋은 학습 및 평가 과정에서 활용할 수 없으며, 모든 데이터 사용 내역은 투명하게 공개되어야 한다.

작은 규모의 모델로도 수능형 문제에서 높은 성능을 달성하며, 대형 언어 모델(GPT, Claude, Gemini)과 비교해 경쟁력 있는 수능 특화 AI 모델을 개발하는 것이 목표이다.
<br />

**1.2 평가지표**

평가 지표는 정확도(Accuracy)이며, 모델이 맞춘 문제 수 / 전체 문제 수로 계산한다.

<img width="700" alt="평가지표" src="https://github.com/user-attachments/assets/9bae7ae7-cdb4-44ea-8844-ccfe9a5d09f0">

<br />

## 2 프로젝트 팀 구성 및 역할

## 팀원 소개

| **이름** | **프로필** | **역할** | **GitHub** |
| --- | --- | --- | --- |
| **강정완** | <img alt="강정완" width="140" height="140" src="https://github.com/user-attachments/assets/4f48f414-1da1-4476-acfa-b73104604db7" /> | - Streamlit Data Viewer 제작 <br /> - 시험지 PDF 데이터 추출 <br /> - OpenAI API 모듈 구현 <br /> - 합성 데이터 생성 모듈 구현 <br /> - 합성 데이터(문제, 문제풀이) 생성 <br /> - 데이터 정리 <br /> - 모델 학습 | [GJ98](https://github.com/GJ98) |
| **김민선** | <img alt="김민선" width="140" height="140" src="https://github.com/user-attachments/assets/603a2aaa-58ea-416e-b366-097f845bf5d5" /> | - 프로젝트 협업 관리(깃허브 이슈 템플릿 및 pre-commit 설정, Commitizen <br /> 설정, 노션 관리) <br /> - 데이터 분석 <br /> - 시험지 PDF 데이터 추출 <br /> - Claude API 활용 모듈 구현 <br /> - RAG를 위한 Media Wiki API, Langchain Wikipedia Retriever 활용 모듈 구현 | [CLM-BONNY](https://github.com/CLM-BONNY) |
| **서선아** | <img alt="서선아" width="140" height="140" src="https://github.com/user-attachments/assets/57c9c737-28d7-4ed0-b8c9-48eb5daaeb8a" /> | - open-ko-llm-leaderboard 모델 조사 <br /> - 시험지 PDF 데이터 추출 <br /> - RAG 작동방식/모델/데이터 조사 <br /> - RAG 데이터 키워드 추출 프롬프트 엔지니어링 | [seon03](https://github.com/seon03) |
| **이인구** | <img alt="이인구" width="140" height="140" src="https://github.com/user-attachments/assets/51a26c46-03e5-4a42-94de-9f7f3034383c" /> | - 시험지 PDF 데이터 추출 <br /> - RAG 데이터 조사 | [inguelee](https://github.com/inguelee) |
| **이재협** | <img alt="이재협" width="140" height="140" src="https://github.com/user-attachments/assets/75b8ef71-6afe-4654-9843-a0913937a1de" /> | - 프로젝트 코드 템플릿 구성 <br /> - 시험지 PDF 데이터 추출 <br /> - 모델링 관련 모듈 구현(SFT, CoT, DPO) <br /> - 학습/추론 코드 구현 <br /> - Unsloth 라이브러리 도입 | [jhyeop](https://github.com/jhyeop) |
| **임상엽** | <img alt="임상엽" width="140" height="140" src="https://github.com/user-attachments/assets/2d66dd95-8c1c-4441-9511-6bf2fc3b06170" /> | - 작업큐 구현 및 개선 <br /> - Swarm을 사용한 Agent Framework 구현 <br /> - Unsloth 라이브러리 도입 | [gityeop](https://github.com/gityeop) |

<br />

## 3 프로젝트

**3.1 프로젝트 진행 일정**

- 프로젝트는 EDA, 데이터 전처리, 데이터 증강, 데이터 확장, SFT, CoT, DPO, RAG 순서로 진행했다.
<img width="900" alt="프로젝트 일정" src="https://github.com/user-attachments/assets/c606e07f-48bd-45b8-988e-5e245e5d3309">

<br />
<br />

**3.2 프로젝트 폴더 구조**

```

├── EDA/                                      # EDA 관련 파일
├── config/
│   ├── config.yaml                           # model, dataset 등 설정 파일
├── modules/
│   ├── data_module.py                        # data 클래스
│   ├── generate_prompt.py                    # 프롬프트 템플릿 적용 데이터셋 생성 함수
│   ├── model.py                              # 모델 클래스
│   ├── trainer.py                            # trainer 클래스
│   ├── extract_rag_keyword.py                # 추론 입력 데이터 과목 분류 및 키워드 추출 클래스
│   ├── retrieval_wikipedia_document.py       # 키워드 기반 위키피디아 문서 검색 함수
│   └── batch_infrence_swarm.py               # Swarm 활용 Agent 추론 함수
├── utils/
│   ├── extract_data/                         # 외부 시험지 데이터 추출 관련 파일
│   ├── job_queue/                            # 작업큐 관련 파일
│   ├── synthetic_data/                       # 합성 데이터 생성 관련 파일					
├── inference.py                              # 추론 파일
├── inference_cot.py                          # CoT 적용 추론 파일
├── inference_cot_batch.py                    # batch 방식 CoT 적용 추론 파일
├── train.py                                  # 모델 학습 파일
└── requirements.txt                          # 필요 패키지 명시 파일

```

<br />

## 4 EDA

**4.1 토큰 개수**
아래 그래프는 문제를 Qwen/Qwen2.5-3B-Instruct 토크나이저로 토큰화했을 때, 토큰 개수의 분포를 나타낸 것이다.

<img width="500" src="https://github.com/user-attachments/assets/f09af710-f4da-4a49-a18b-02e215522666" />

모든 문제의 토큰 개수가 1024개 이하이므로, 학습 시 최대 토큰 길이를 1024개로 설정해도 문제가 없다. 다만, 새롭게 생성된 문제의 토큰 개수가 1024개를 초과할 수 있으므로, 증강된 문제의 토큰 개수 분포를 파악해 학습에서 제외되지 않도록 주의해야 한다.

**4.2 Chain of Thought (CoT) 필요성**

<img width="500" src="https://github.com/user-attachments/assets/d21dfa91-940f-41d9-8916-4ed8e7845a34" />

위 문제 generation-for-nlp-1414와 같이 수능은 복잡한 사고 과정을 필요로 하는 문제가 다수 포함된 시험이다. 이때, 모델이 복잡한 사고 과정을 설명하도록 유도하는 Chain of Thought (CoT) 프롬프트 테크닉을 사용하면 답변 정확도가 올라갈 수 있을거라 판단해 문제풀이를 설명하도록 요구하는 CoT 실험을 진행할 계획이다.
    
<br />

## 5 프로젝트 수행

**5.1 Data Processing**

- Data Preprocessing
- Data Augmentation
    - Extract Exam Data
    - Synthetic Data

**5.2 Modeling**

- SFT
- CoT
- DPO

**5.3 Advanced**

- Agent
- RAG

**5.4 Ensemble**

- Hard Voting

<br />

## 6 Wrap-up Report

자세한 내용은 <a href="https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-04-lv3/blob/main/GenerationforNLP_NLP_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(04%EC%A1%B0).pdf">**Wrap-up Report**</a>를 참고해 주세요 !
