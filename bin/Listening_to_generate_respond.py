# from speech_recognition_module import recognize_speech_from_mic
# from openai_response_module import get_openai_response
# from speech_synthesis_module import synthesize_speech

# # ========================================================================================================
# # 下面是封装好的方法,就是不涉及情绪的识别
# from speech_recognition_module import recognize_speech_from_mic
# from openai_response_module import get_openai_response
# from speech_synthesis_module import synthesize_speech

# def generate_single_respond(role_description="a lovely dog named Miro-e"):
#     """
#     执行一次语音交互，包括语音识别、生成回复并进行语音合成。
    
#     参数：
#         role_description (str): 设定OpenAI角色描述。
#     """
#     user_input = recognize_speech_from_mic()
#     generated_respond_filepath = None
#     if user_input:
#         # 检查是否包含退出命令
#         if user_input.lower() in ['退出', '结束', '停止', 'exit', 'quit', 'stop']:
#             print("已接收到退出命令。")
#             # return False  # 用于指示是否继续进行交互
        
#         # 获取OpenAI的回复
#         response_text = get_openai_response(user_input, role_description=role_description)
#         if response_text:
#             # 合成并播放语音
#             generated_respond_filepath = synthesize_speech(response_text)
#     else:
#         print("未检测到有效输入，请重试。")
    

#     # return True  # 用于指示继续进行交互
#     return generated_respond_filepath

# # ========================================================================================================


from speech_recognition_module import recognize_speech_from_mic
from openai_response_module import get_openai_response
from speech_synthesis_module import synthesize_speech

def generate_single_respond(role_description="a lovely dog named Miro-e", detected_emotion='neutral', user_input = None):
    """
    执行一次语音交互，包括语音识别、生成回复并进行语音合成。
    
    参数：
        role_description (str): 设定OpenAI角色描述。
        detected_emotion (str): 用于输入检测到的用户的emotion
        user_input: 用户的输入，默认为空，如果没有传入则会从麦克风重新读取。但理论上应该已经有了
    """
    if user_input is None:
        user_input = recognize_speech_from_mic()
    generated_respond_filepath = None
    if user_input:
        # 检查是否包含退出命令
        if user_input.lower() in ['退出', '结束', '停止', 'exit', 'quit', 'stop']:
            print("已接收到退出命令。")
            # return False  # 用于指示是否继续进行交互
        
        # 获取OpenAI的回复
        response_text = get_openai_response(user_input, role_description=role_description, users_emotion=detected_emotion)
        if response_text:
            # 合成并播放语音
            generated_respond_filepath = synthesize_speech(response_text, style = detected_emotion)
    else:
        print("未检测到有效输入，请重试。")
    

    # return True  # 用于指示继续进行交互
    return generated_respond_filepath






if __name__ == "__main__":
    generate_single_respond(user_input='Hi, how are you?', detected_emotion='neutral')


