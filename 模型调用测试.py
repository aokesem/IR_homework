from openai import OpenAI

client = OpenAI(api_key="sk-ipf4pyozQpL905IkkcEe0nycy1r1xEr0oYSAgpf7IW9o0F6b", base_url="https://tb.api.mkeai.com/v1") 

response = client.chat.completions.create(
    model="deepseek-r1-0528",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，你是谁"},
    ],
    stream=False
)

print(response.choices[0].message.content)