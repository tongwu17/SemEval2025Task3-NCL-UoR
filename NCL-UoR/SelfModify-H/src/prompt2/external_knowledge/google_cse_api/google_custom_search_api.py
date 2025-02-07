import json
import ast
import requests
from goose3 import Goose
from requests.exceptions import SSLError, RequestException


# 定义 Google Custom Search API 的参数
API_KEY = "AIzaSyBVZfk7xkrKW9pi2dzYjbe0nua2TQKSeJ0"  # 替换为您的 Google API 密钥
CX = "c08cbf9c245c44720"  # 替换为您的搜索引擎 CX 值

# 初始化 Goose3
g = Goose()

# 定义 Google 搜索函数
def google_search(query, lang, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CX,
        "q": query,
        "hl": lang,  # 设置查询语言
        "num": num_results
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("items", [])
        return [item["link"] for item in results]
    except SSLError as ssl_err:
        print(f"SSL Error occurred while querying '{query}': {ssl_err}")
        return []
    except RequestException as req_err:
        print(f"Request Error occurred while querying '{query}': {req_err}")
        return []

# 使用 Goose3 解析网页内容
def parse_webpage(url):
    try:
        article = g.extract(url=url)
        return article.title, article.cleaned_text
    except Exception as e:
        print(f"Failed to parse {url}: {e}")
        return None, None

# 读取 CSV 文件并处理
def process_csv(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:

        for line in infile:
            row = json.loads(line)
            lang = row.get("lang")
            keyphrases = row.get("keywords", [])

            if lang and keyphrases:
                links = []
                parsed_content = []

                for keyphrase in keyphrases:
                    query_links = google_search(keyphrase, lang)
                    links.extend(query_links)

                    for link in query_links:
                        title, text = parse_webpage(link)
                        if title and text:
                            parsed_content.append(f"{title}: {text}")

                row["google_links"] = links
                row["parsed_content"] = "".join(parsed_content)
            else:
                row["google_links"] = []
                row["parsed_content"] = []

            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    # files = ['mushroom.ca-tst.v1.jsonl',
    # for file in files:
    input_file = f'data/hong/method_1/keywords/test/mushroom.fi-tst.v1.jsonl'
    output_file = f'data/hong/method_1/context/test/mushroom.fi-tst.v1.jsonl'
    process_csv(input_file, output_file)
    print(f"处理完成，结果已保存到 {output_file}")