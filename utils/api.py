import requests
import xml.etree.ElementTree as ET
from typing import Dict, Union, List

class ArxivApi:

    def __init__(self) -> None:
        self.base_url = "http://export.arxiv.org/api"
        self.session = requests.Session()
        self.namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
    
    def _get_query_params(self, search_query: str = None, id_list: str = None, start: int = 0, max_results: int = 10) -> Dict[str, Union[str, int]]:
        params = {}

        if search_query:
            params['search_query'] = search_query
        
        if id_list:
            params['id_list'] = id_list
        
        params['start'] = start
        params['max_results'] = max_results

        return params
    
    def _parse_query_xml(self, xml_content: bytes) -> List[Dict]:
        root = ET.fromstring(xml_content)
        entries = root.findall('atom:entry', self.namespaces)

        results = []
        for entry in entries:
            result = {
                'id': entry.find('atom:id', self.namespaces).text,
                'updated': entry.find('atom:updated', self.namespaces).text,
                'published': entry.find('atom:published', self.namespaces).text,
                'title': entry.find('atom:title', self.namespaces).text,
                'summary': entry.find('atom:summary', self.namespaces).text.strip(),
                'authors': [author.find('atom:name', self.namespaces).text for author in entry.findall('atom:author', self.namespaces)],
                'link': entry.find('atom:link[@rel="alternate"]', self.namespaces).get('href'),
                'pdf_link': entry.find('atom:link[@title="pdf"]', self.namespaces).get('href'),
                'primary_category': entry.find('arxiv:primary_category', self.namespaces).get('term'),
                'categories': [category.get('term') for category in entry.findall('atom:category', self.namespaces)]
            }
            results.append(result)
        return results


    
    def query(self, search_query: str = None, id_list: str = None, start: int = 0, max_results: int = 10):
        url = f"{self.base_url}/query"
        params = self._get_query_params(search_query, id_list, start, max_results)
        try:
            response = self.session.get(url=url, params=params)
            response.raise_for_status()
            return self._parse_query_xml(response.content)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None


if __name__ == "__main__":
    api = ArxivApi()
    print(api.query('attention is all you need', max_results=1))