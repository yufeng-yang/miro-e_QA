# import azure.cognitiveservices.speech as speechsdk

# def synthesize_speech(text, style='cheerful', rate='0%', pitch='10%', filename="output1.wav"):
    # 从环境变量中读取密钥
    # speech_key = os.getenv("AZURE_SPEECH_KEY")
    # service_region = "uksouth"

    # if speech_key is None:
    #     raise ValueError("Azure Speech key not set. Please define AZURE_SPEECH_KEY as an environment variable.")
#     speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
#     speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"

#     # 指定输出文件
#     audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)

#     # 创建语音合成器
#     speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

#     # 创建 SSML 文本以控制语音特性
#     ssml_text = f"""
#     <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
#            xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
#         <voice name='en-US-AriaNeural'>
#             <mstts:express-as style='{style}'>
#                 <prosody rate='{rate}' pitch='{pitch}'>{text}</prosody>
#             </mstts:express-as>
#         </voice>
#     </speak>
#     """

#     try:
#         result = speech_synthesizer.speak_ssml_async(ssml_text).get()
#         if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
#             print(f"语音合成完成，已保存到 '{filename}'。")
#         elif result.reason == speechsdk.ResultReason.Canceled:
#             cancellation_details = result.cancellation_details
#             print(f"语音合成被取消: {cancellation_details.reason}")
#             if cancellation_details.reason == speechsdk.CancellationReason.Error:
#                 print(f"错误详情: {cancellation_details.error_details}")
#     except Exception as e:
#         print(f"语音合成时出错: {e}")

# 策略：！！！为什么不干脆从源头开始，最开始生成的就是8000sample_rate的单声道文件不就避免后续的问题了。
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
import os

def synthesize_speech(text, style='cheerful', rate='0%', pitch='10%', filename="output1.wav"):
    # 设置 Azure 语音服务密钥和区域
    # 从环境变量中读取密钥
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    service_region = "uksouth"

    if speech_key is None:
        raise ValueError("Azure Speech key not set. Please define AZURE_SPEECH_KEY as an environment variable.")
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"

    # 指定输出临时文件
    temp_filename = "temp_output.wav"
    audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)

    # 创建语音合成器
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # 创建 SSML 文本以控制语音特性
    ssml_text = f"""
    <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
           xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
        <voice name='en-US-AriaNeural'>
            <mstts:express-as style='{style}'>
                <prosody rate='{rate}' pitch='{pitch}'>{text}</prosody>
            </mstts:express-as>
        </voice>
    </speak>
    """

    try:
        result = speech_synthesizer.speak_ssml_async(ssml_text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"语音合成完成，已保存到 '{temp_filename}'。")

            # 使用 pydub 将音频格式转换为 8000 Hz、单声道
            sound = AudioSegment.from_wav(temp_filename)
            sound = sound.set_channels(1).set_frame_rate(8000)
            sound.export(filename, format="wav")
            print(f"已转换为 8000 Hz 单声道并保存到 '{filename}'。")

            # 删除临时文件
            os.remove(temp_filename)
            print(f"已删除临时文件 '{temp_filename}'。")
            return filename  # 返回最终文件路径
            
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"语音合成被取消: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"错误详情: {cancellation_details.error_details}")
    except Exception as e:
        print(f"语音合成时出错: {e}")
