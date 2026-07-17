from ollama import chat

response = chat(
    model='llama3.2:1b',
    messages=[
        {
            'role': 'user',
            'content': 'Machine Learning ko simple Hindi me samjhao'
        }
    ]
)

print(response['message']['content'])