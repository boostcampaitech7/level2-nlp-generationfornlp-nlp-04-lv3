import requests
from langchain_community.retrievers import WikipediaRetriever


# MediaWiki API 활용 타이틀 검색 함수
def search_title(title, language="ko"):
    base_url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",  # title 검색
        "titles": title,  # title 지정
        "prop": "extracts",  # 전체 텍스트
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
                "content": page_data.get("extract", "No content available."),
                "url": page_data.get(
                    "fullurl",
                    f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                ),
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
def main(titles, language="ko", load_max_docs=3):
    """
    1. 입력받은 타이틀 리스트의 요소에 대해 타이틀 기반 문서 검색
    2. 타이틀이 일치하는 문서가 없을 경우 키워드 검색으로 대체
    """
    for idx, title in enumerate(titles, start=1):
        print(f"\n=== Processing Title {idx}/{len(titles)}: {title} ===")
        result = search_title(title, language)

        if result is None:
            print(f"'{title}' 문서를 찾을 수 없습니다.")
            keyword_result = search_keyword(title, language, load_max_docs)

            if isinstance(keyword_result, dict) and "error" in keyword_result:
                print(f"키워드 검색 실패: {keyword_result['error']}")
            else:
                print("키워드 검색 결과:")
                for i, doc in enumerate(keyword_result, start=1):
                    print(f"Document {i}:")
                    print(f"Title: {doc.metadata.get('title', 'No Title')}")
                    print(f"Content: {doc.page_content}\n")
        elif "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Page ID: {result['page_id']}")
            print(f"Title: {result['title']}")
            print(f"Content: {result['content']}")


if __name__ == "__main__":
    # 검색할 타이틀 리스트
    titles = ["커피", "주식", "비트코인"]

    # 메인 실행
    main(titles)
