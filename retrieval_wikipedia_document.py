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
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":  # 문서가 존재하지 않을 경우
                return {"error": f"No article found for title '{title}'"}
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
    retriever = WikipediaRetriever(lang=language, load_max_docs=load_max_docs)

    docs = retriever.invoke(keyword)
    return docs


if __name__ == "__main__":
    # 검색할 문서 타이틀
    search_term = "주식"

    # 문서 반환
    result = search_title(search_term)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Title: {result['title']}")
        print(f"Page ID: {result['page_id']}")
        print(f"Content: {result['content']}")
        print(f"URL: {result['url']}")

    # 검색할 키워드
    keyword = "주식"

    # 문서 반환
    results = search_keyword(keyword)

    for doc in results:
        print(doc.metadata["title"])
        print(doc.metadata["summary"])
