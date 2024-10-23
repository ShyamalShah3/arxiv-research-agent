from __future__ import annotations
import os
import json
import requests
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from typing import TypedDict, List
import google.generativeai as gemini
import cohere

load_dotenv()

class SearchQueryGenerationResponse(TypedDict):
    search_query: str
    explanation: str

class ArxivApiQueryResponse(TypedDict):
    id: str
    updated: str
    published: str
    title: str
    summary: str
    authors: List[str]
    link: str
    pdf_link: str
    primary_category: str
    categories: List[str]

class ArxivApiResponse(ArxivApiQueryResponse):
    relevance_score: float

    @classmethod
    def from_query_response(cls, query_response: ArxivApiQueryResponse, relevance_score: float) -> ArxivApiResponse:
        return cls(
            id=query_response["id"],
            updated=query_response["updated"],
            published=query_response["published"],
            title=query_response["title"],
            summary=query_response["summary"],
            authors=query_response["authors"],
            link=query_response["link"],
            pdf_link=query_response["pdf_link"],
            primary_category=query_response["primary_category"],
            categories=query_response["categories"],
            relevance_score=relevance_score
        )


class ArxivApi:

    def __init__(self) -> None:
        self.__base_url = "http://export.arxiv.org/api"
        self.__session = requests.Session()
        self.__namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        gemini.configure(api_key=os.environ['GEMINI_API_KEY'])
        self.__gemini_model = gemini.GenerativeModel('gemini-1.5-flash')
        self.__cohere = cohere.Client(os.getenv('COHERE_API_KEY'))
    
    def __parse_search_query_response(self, response: str) -> SearchQueryGenerationResponse:
        try:
            response_dict = json.loads(response)

            if not all(key in response_dict for key in ('search_query', 'explanation')):
                raise KeyError("Missing required fields in response")
            
            return SearchQueryGenerationResponse(
                search_query=response_dict['search_query'],
                explanation=response_dict['explanation']
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON string: {str(e)}", e.doc, e.pos)
    
    def __get_search_query(self, query: str) -> str:
        prompt = f"""
        You are tasked with generating an optimized search query for the arXiv API based on an input query. Your goal is to create a search query that adheres to the API specifications and is optimized for search engines, particularly focusing on relevant keywords.

        Here are the key points from the arXiv API specifications to keep in mind:

        Field prefixes:

            ti: Title

            au: Author

            abs: Abstract

            co: Comment

            jr: Journal Reference

            cat: Subject Category

            rn: Report Number

            all: All of the above

        Boolean operators:

            AND

            OR

            ANDNOT

        Grouping operators:

            Parentheses: Use %28 for ( and %29 for )

            Double quotes: Use %22 for both opening and closing quotes

            Space: Use + to represent a space in the URL

        To construct the search query:

            Analyze the input query to identify key concepts, authors, or specific phrases.

            Use appropriate field prefixes to target specific areas of the articles.

            Combine search terms using Boolean operators.

            Group terms or phrases using parentheses or double quotes as needed.

        When handling special characters and spaces:

            Replace spaces with + signs.

            Encode parentheses as %28 and %29.

            Encode double quotes as %22.

            Ensure all other special characters are properly URL-encoded.

        To optimize the search query:

            Prioritize the most relevant keywords from the input query.

            Use specific field prefixes when possible to narrow down results.

            Combine related concepts with the OR operator to broaden the search when appropriate.

            Use the AND operator to ensure all important concepts are included.

            Consider using the ANDNOT operator to exclude irrelevant results if necessary.

        Now, consider the following input query:
        <input_query>
        {query}
        </input_query>

        Based on this input query, generate an optimized search query for the arXiv API. Ensure that your query is properly formatted and URL-encoded.

        Provide your final output in the following format:
        <search_query>
        Your generated search query here
        </search_query>
        <explanation>
        A brief explanation of your query construction and optimization choices
        </explanation>
        """
        response = self.__gemini_model.generate_content(prompt, generation_config=gemini.GenerationConfig(
            response_mime_type='application/json',
            response_schema=SearchQueryGenerationResponse
        ))
        return self.__parse_search_query_response(response.text)['search_query']
    
    def __get_query_url(self, search_query: str, start: int = 0, max_results: int = 10) -> str:
        if not search_query:
            raise ValueError("Error: Empty search_query")
        
        params = []
        params.append(f"search_query={search_query}")
        params.append(f"start={start}")
        params.append(f"max_results={max_results}")
        
        query_string = "&".join(params)
        return f"{self.__base_url}/query?{query_string}"
    
    def __parse_query_xml(self, xml_content: bytes) -> List[ArxivApiQueryResponse]:
        root = ET.fromstring(xml_content)
        entries = root.findall('atom:entry', self.__namespaces)

        results = []
        for entry in entries:
            result = ArxivApiQueryResponse(
                id=entry.find('atom:id', self.__namespaces).text,
                updated=entry.find('atom:updated', self.__namespaces).text,
                published=entry.find('atom:published', self.__namespaces).text,
                title=entry.find('atom:title', self.__namespaces).text,
                summary=entry.find('atom:summary', self.__namespaces).text.strip(),
                authors=[author.find('atom:name', self.__namespaces).text for author in entry.findall('atom:author', self.__namespaces)],
                link=entry.find('atom:link[@rel="alternate"]', self.__namespaces).get('href'),
                pdf_link=entry.find('atom:link[@title="pdf"]', self.__namespaces).get('href'),
                primary_category=entry.find('arxiv:primary_category', self.__namespaces).get('term'),
                categories=[category.get('term') for category in entry.findall('atom:category', self.__namespaces)]
            )
            results.append(result)
        return results
    
    def __rerank_results(self, query: str, documents: List[ArxivApiQueryResponse], top_n: int = 10) -> List[ArxivApiResponse]:
        rank_fields = ['summary']
        rerank_results = self.__cohere.rerank(query=query, documents=documents, rank_fields=rank_fields, model='rerank-english-v3.0', top_n=top_n)
        results: List[ArxivApiResponse] = []
        for result in rerank_results.results:
            doc = documents[result.index]
            results.append(
                ArxivApiResponse.from_query_response(doc, result.relevance_score)
            )
        return results
    
    def query(self, query: str, start: int = 0, max_results: int = 100, top_n: int = 100) -> List[ArxivApiResponse]:
        try:
            url = self.__get_query_url(self.__get_search_query(query), start, max_results)
            response = self.__session.get(url=url)
            response.raise_for_status()
            return self.__rerank_results(query, self.__parse_query_xml(response.content), top_n)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None


if __name__ == "__main__":
    api = ArxivApi()
    query = 'Retrieval Augmented Generation'
    docs = api.query(query, max_results=100, top_n=10)
    print(docs)
