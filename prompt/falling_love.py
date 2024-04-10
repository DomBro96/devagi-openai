import os
from openai import OpenAI

# 加载 .env 到环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 配置 OpenAI 服务

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

response = client.chat.completions.create(
    messages=[
         {
            "role": "system",
            "content": "你是一个身材很好的男明星，你的名字叫于适。你很喜欢粉丝对你的爱，你也很爱他们。",
        },

        {
            "role": "user",
            "content": "我是你的粉丝，我可以摸摸你的肌肉吗？",
        }
    ],
    model="gpt-3.5-turbo",
)

print(response.choices[0].message)

