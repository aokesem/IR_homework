from openai import OpenAI

client = OpenAI(api_key="sk-ipf4pyozQpL905IkkcEe0nycy1r1xEr0oYSAgpf7IW9o0F6b", base_url="https://tb.api.mkeai.com/v1") # base_url后面必须加上/v1

response = client.chat.completions.create(
    model="deepseek-r1-0528", # 可以替换为其他可用模型
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你能做什么"},
    ],
    stream=False
)

print(response.choices[0].message.content)