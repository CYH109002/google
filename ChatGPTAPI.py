from openai import OpenAI
client = OpenAI()

def Connet_ChatGPT(fruit_name):
    print("")
    messages = []
    message = f"what is {fruit_name}"
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    desired_message = response.choices[0].message.content
    print(f"This is {fruit_name}")
    print(f"ChatGPT:{desired_message}")