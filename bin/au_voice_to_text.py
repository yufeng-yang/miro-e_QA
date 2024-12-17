import azure.cognitiveservices.speech as speechsdk
import os
class AudioFileRecognizer:
    def __init__(self, audio_file_path):
        """
        初始化录音文件识别器。
        :param audio_file_path: 音频文件的路径
        """
        self.audio_file_path = audio_file_path
        self.speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"),  # 替换为你的 Azure 密钥
            region="uksouth"  # 替换为你的 Azure 语音服务区域
        )
        self.speech_config.speech_recognition_language = "en-US"  # 设置识别语言

    def recognize_from_audio_file(self):
        """
        使用 Azure Speech-to-Text 将音频文件转为文本。
        """
        # 使用音频文件作为输入
        audio_input = speechsdk.AudioConfig(filename=self.audio_file_path)
        
        # 创建识别器
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_input
        )
        
        # 开始识别
        print("正在识别录音文件内容...")
        result = speech_recognizer.recognize_once()  # 处理整个音频文件
        
        # 检查结果
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"识别成功：{result.text}")
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("未能识别任何内容。")
            return None
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"识别被取消：{cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"错误详细信息：{cancellation_details.error_details}")
            return None


if __name__ == "__main__":
    # 替换为你的录音文件路径
    audio_file_path = "/home/yufeng/mdk/output1.wav"
    
    # 创建识别器实例
    recognizer = AudioFileRecognizer(audio_file_path)
    
    # 执行识别
    text = recognizer.recognize_from_audio_file()
    if text:
        print(f"录音文本内容：{text}")
