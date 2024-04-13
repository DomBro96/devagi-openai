import numpy as np
from numpy import dot
from numpy.linalg import norm

from openai import OpenAI
import os
# 加载环境变量
from dotenv import load_dotenv, find_dotenv
from es_engine import search

_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY


client = OpenAI()


def cos_sim(a, b):
    '''余弦距离 -- 越大越相似'''
    return dot(a, b)/(norm(a)*norm(b))


def l2(a, b):
    '''欧式距离 -- 越小越相似'''
    x = np.asarray(a)-np.asarray(b)
    return norm(x)

def get_embeddings(texts, model="text-embedding-ada-002",dimensions=None):
    '''封装 OpenAI 的 Embedding 模型接口'''
    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

test_query = ["测试文本"]
vec = get_embeddings(test_query)[0]
print(vec[:10])
print(len(vec))

# 且能支持跨语言
# 国际冲突
query = "global conflicts"

documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]


query_vec = get_embeddings([query])[0]
doc_vecs = get_embeddings(documents)

print("Cosine distance:")
print(cos_sim(query_vec, query_vec))
for vec in doc_vecs:
    print(cos_sim(query_vec, vec))


'''
Cosine distance:
1.0
0.7622376995981268 
0.7564484035029613 
0.7426558372998222 
0.7077987135264396 
0.7254230492369406
'''

print("\nEuclidean distance:")
print(l2(query_vec, query_vec))
for vec in doc_vecs:
    print(l2(query_vec, vec))

'''
Euclidean distance:
0.0
0.6895829747071623
0.6979278474290739
0.7174178392255148
0.7644623330084925
0.7410492267209755
'''