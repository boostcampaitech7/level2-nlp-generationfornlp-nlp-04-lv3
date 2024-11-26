import os
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_community.retrievers import WikipediaRetriever


# MediaWiki API 활용 타이틀 검색 함수
def search_title(title, language="ko"):
    base_url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",  # title 검색
        "titles": title,  # title 지정
        "prop": "extracts",  # 전체 텍스트
        "prop": "pageprops|extracts",  # pageprops에서 설명 정보 가져옴
        "exintro": True,  # 첫 번째 섹션만 가져오기
        "explaintext": True,  # 텍스트 형식으로 요청
        "inprop": "url",  # URL 정보 포함
        "format": "json",  # JSON 포맷으로 반환
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # HTTP 에러 확인
        data = response.json()

        # 결과 처리
        pages = data.get("query", {}).get("pages", {})  # 검색 수행
        for page_id, page_data in pages.items():
            if page_id == "-1":  # 문서가 존재하지 않을 경우
                return None
            return {
                "title": page_data.get("title", "Unknown"),
                "page_id": page_id,
                "summary": page_data.get(
                    "extract", "No summary available."
                ),  # 요약 가져오기
            }
    except requests.RequestException as e:
        return {"error": f"API request failed: {e}"}


# Langchain Wikipedia Retriever 활용 키워드 검색 함수
def search_keyword(keyword, language="ko", load_max_docs=3):
    try:
        retriever = WikipediaRetriever(lang=language, load_max_docs=load_max_docs)
        docs = retriever.invoke(keyword)  # 검색 수행
        if not docs:  # 결과가 비어 있을 경우
            return {"error": f"No results found for keyword '{keyword}'"}
        return docs
    except Exception as e:  # 예외 처리
        return {"error": f"An error occurred: {e}"}


# 메인 retrieval 로직 함수
def main(titles, language="ko", load_max_docs=1):
    """
    1. 입력받은 타이틀 리스트의 요소에 대해 타이틀 기반 문서 검색
    2. 타이틀이 일치하는 문서가 없을 경우 키워드 검색으로 대체
    """
    retrieved_texts = []

    for idx, title in enumerate(titles, start=1):
        print(f"\n=== Processing Title {idx}/{len(titles)}: {title} ===")
        result = search_title(title, language)

        if result is None:
            print(f"'{title}' 문서를 찾을 수 없습니다.")
            keyword_result = search_keyword(title, language, load_max_docs)
            if isinstance(keyword_result, dict) and "error" in keyword_result:
                print(f"키워드 검색 실패: {keyword_result['error']}")
                retrieved_texts.append(None)
            else:
                combined_text = "\n".join(
                    [
                        doc.metadata.get("summary", "No Summary")
                        for doc in keyword_result
                    ]
                )
                retrieved_texts.append(combined_text)
        elif "error" in result:
            print(f"Error: {result['error']}")
            retrieved_texts.append(None)
        else:
            retrieved_texts.append(result["summary"])
    return retrieved_texts


if __name__ == "__main__":
    load_dotenv()

    # 데이터 로드
    file_path = os.path.join(
        os.getenv("ROOT_DIR"), "data/validation_keyword_qwen_three.csv"
    )
    data = pd.read_csv(file_path)

    # is_social이 1인 경우 keywords 추출 및 main 실행
    data["document"] = data.apply(
        lambda row: (
            main(eval(row["keywords"]), "ko") if row["is_social"] == 1 else None
        ),
        axis=1,
    )

    # 결과 저장
    output_path = os.path.join(
        os.getenv("ROOT_DIR"), "data/validation_keyword_qwen_three_2.csv"
    )
    data.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
