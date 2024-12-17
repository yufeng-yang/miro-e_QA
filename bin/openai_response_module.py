# from openai import OpenAI

# def get_openai_response(prompt):
#     client = OpenAI()
#     try:
#         completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#                     # 这是可以用来调试角色
#                     {"role": "system", "content": "You are a lovely dog. Your name is Miro-e. You pretend my text input is voice input"},
#                     {"role": "user", "content": prompt}
#                 ]
#         )

#         response_text = completion.choices[0].message.content

#         print(f"Miro-e: {response_text}")
#         return response_text
#     except Exception as e:
#         print(f"请求 OpenAI API 时出错: {e}")
#         return None


from openai import OpenAI

def get_openai_response(prompt, role_description="a lovely dog named Miro-e", users_emotion='neutral'):
    client = OpenAI()
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                # 动态设置系统角色
                {"role": "system", "content": f"You are {role_description}. User's emotion is {users_emotion}. You pretend my text input is voice input, and Chrismas is coming soon. Finish with question. Give me maximum 4 sentences."},
                {"role": "user", "content": prompt}
            ]
        )

        response_text = completion.choices[0].message.content

        print(f"Role ({role_description}): {response_text}")
        return response_text
    except Exception as e:
        print(f"请求 OpenAI API 时出错: {e}")
        return None



