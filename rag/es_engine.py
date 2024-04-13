from elasticsearch7 import Elasticsearch, helpers
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from  es_init import to_keywords
from pdf_helper import extract_text_from_pdf


# 1. 创建Elasticsearch连接
es = Elasticsearch(
    hosts=['http://117.50.198.53:9200'],  # 服务地址与端口
    http_auth=("elastic", "FKaB1Jpz0Rlw0l6G"),  # 用户名，密码
)

# 2. 定义索引名称
index_name = "teacher_demo_index123"

# 3. 如果索引已存在，删除它（仅供演示，实际应用时不需要这步）
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# 4. 创建索引
es.indices.create(index=index_name)

paragraphs = extract_text_from_pdf("rag\llama2.pdf", min_line_length=10)
# 5. 灌库指令
actions = [
    {
        "_index": index_name,
        "_source": {
            "keywords": to_keywords(para),
            "text": para
        }
    }
    for para in paragraphs
]

# 6. 文本灌库
helpers.bulk(es, actions)

def search(query_string, top_n=3):
    # ES 的查询语言
    search_query = {
        "match": {
            "keywords": to_keywords(query_string)
        }
    }
    res = es.search(index=index_name, query=search_query, size=top_n)
    return [hit["_source"]["text"] for hit in res["hits"]["hits"]]


results = search("how many parameters does llama 2 have?", 2)
for r in results:
    print(r+"\n")


# results2 和 result3 有相同语义，但得到的予以结果是不相关的...
results2 = search("Does llama 2 have a chat version?", 2)   
print("Does llama 2 have a chat version? \n")
for r in results2:
    print(r+"\n")

print("Does llama 2 have a conversational variant? \n")
results3 = search("Does llama 2 have a conversational variant?", 2)
for r in results3:
    print(r+"\n")

