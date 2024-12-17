# import speech_recognition as sr

# class KeywordRecognizer:
#     def __init__(self):
#         self.recognizer = sr.Recognizer()
#         self.microphone = sr.Microphone()

#         # 标志位，确保每个事件只触发一次
#         self.method1_triggered = False
#         self.method2_triggered = False

#     def recognize_and_trigger(self):
#         """
#         从麦克风获取语音输入并识别关键词，触发对应的方法。
#         """
#         with self.microphone as source:
#             print("请说话...")
#             # 降低环境噪音
#             self.recognizer.adjust_for_ambient_noise(source)
#             # 获取音频数据
#             audio = self.recognizer.listen(source)

#         try:
#             # 识别语音内容
#             text = self.recognizer.recognize_google(audio, language="en-US").lower()
#             print(f"你说: {text}")

#             # 根据关键词触发方法
#             if "hey lovely dog" in text and not self.method1_triggered:
#                 self.method1()
#                 self.method1_triggered = True  # 设置方法1为已触发

#             elif "goodbye miro" in text and not self.method2_triggered:
#                 self.method2()
#                 self.method2_triggered = True  # 设置方法2为已触发

#         except sr.UnknownValueError:
#             print("抱歉，无法理解你的话。")
#         except sr.RequestError:
#             print("无法连接到语音识别服务。")

#     def method1(self):
#         """
#         方法1：定义在识别到 'hey, miro' 后执行的逻辑。
#         """
#         print("方法1已触发！执行相关操作...")

#     def method2(self):
#         """
#         方法2：定义在识别到 'goodbye, miro' 后执行的逻辑。
#         """
#         print("方法2已触发！执行相关操作...")


# if __name__ == "__main__":
#     # 创建关键词识别器
#     recognizer = KeywordRecognizer()

#     while True:
#         print("正在监听，请说话 (按 Ctrl+C 退出)...")
#         recognizer.recognize_and_trigger()


# import azure.cognitiveservices.speech as speechsdk
# import os
# def recognize_keywords():
#     # 配置 Speech 服务
#     speech_config = speechsdk.SpeechConfig(subscription = os.getenv("AZURE_SPEECH_KEY"), region="uksouth")
#     speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

#     # 定义事件触发标志
#     triggered_flags = {"hey miro": False, "goodbye miro": False}

#     def process_text(evt):
#         """
#         处理语音识别结果并触发事件。
#         """
#         text = evt.result.text.lower()
#         print(f"you said: {text}")

#         if "hey miro" in text and not triggered_flags["hey miro"]:
#             print("触发事件1：你好，Miro！")
#             triggered_flags["hey miro"] = True

#         elif "goodbye miro" in text and not triggered_flags["goodbye miro"]:
#             print("触发事件2：再见，Miro！")
#             triggered_flags["goodbye miro"] = True

#     # 注册事件回调
#     speech_recognizer.recognized.connect(process_text)

#     # 开始监听
#     print("开始监听，请说 'hey miro' 或 'goodbye miro'...")
#     speech_recognizer.start_continuous_recognition()

#     try:
#         while True:
#             pass
#     except KeyboardInterrupt:
#         print("停止监听。")
#         speech_recognizer.stop_continuous_recognition()

# if __name__ == "__main__":
#     recognize_keywords()

# ========================================================================================================
# V3:  Azure with 字典格式的关键词触发器

import azure.cognitiveservices.speech as speechsdk
import os


class KeywordRecognizer:
    def __init__(self, keyword_actions):
        """
        初始化关键词识别器。
        :param keyword_actions: 关键词及其对应方法和是否需要重置的标志，例如：
                                {"hey miro": {"action": self.action1, "reset": False}, ...}
        """
        self.speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region="uksouth")
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config)

        # 存储关键词及其配置（方法和重置标志）
        self.keyword_actions = keyword_actions

        # 初始化触发标志
        self.triggered_flags = {keyword: False for keyword in keyword_actions}

    def process_text(self, evt):
        """
        处理语音识别结果并根据关键词触发对应方法。
        """
        text = evt.result.text.lower()
        print(f"You said: {text}")

        for keyword, config in self.keyword_actions.items():
            action = config["action"]
            reset = config["reset"]

            # 检查关键词是否匹配且未被触发，触发方法并设置标志
            if keyword in text and not self.triggered_flags[keyword]:
                print(f"Triggering action for: {keyword}")
                action()  # 调用对应的方法
                if not reset:
                    self.triggered_flags[keyword] = True  # 仅当无需重置时设置为已触发

    def start_listening(self):
        """
        启动语音识别并监听关键词。
        """
        print("Listening for keywords...")
        self.speech_recognizer.recognized.connect(self.process_text)
        self.speech_recognizer.start_continuous_recognition()

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Stopping...")
            self.speech_recognizer.stop_continuous_recognition()
    
    def stop_listening(self):
        print("Stopping recognition...")
        self.speech_recognizer.stop_continuous_recognition()
        self.speech_recognizer = None

    def reset_flags(self):
        """
        重置需要重置的关键词触发标志。
        """
        for keyword, config in self.keyword_actions.items():
            if config["reset"]:  # 仅重置需要重置的关键词
                self.triggered_flags[keyword] = False
        print("Reset flags for keywords that allow reset.")


# 定义触发方法
def action_hey_miro():
    print("你好，Miro！执行方法1...")


import rospy
from Listening_to_generate_respond import generate_single_respond
from temp_audio_play import AudioPlayback
import miro2 as miro
def action_goodbye_miro():
    print("再见，Miro！执行方法2...")
    # sound_file = generate_single_respond()
    interface = miro.lib.RobotInterface
    playback = AudioPlayback(interface, "/miro", '/home/yufeng/mdk/angry_dog_barking_mono_8000Hz.WAV')
    rospy.sleep(2)
    playback.play()



def starting_voice_control():
    print("Starting voice control...")


# 下面是具体的动作命令，可以多次执行
def action_wagging_your_tail():
    print("坐下！摇尾巴！")
    
def stop_action():
    print("停止！")


# if __name__ == "__main__":
#     # 配置关键词、触发方法和是否需要重置
#     keyword_actions = {
#         "hey buddy": {"action": action_hey_miro, "reset": False},  # 不需要重置
#         "goodbye": {"action": action_goodbye_miro, "reset": False},  # 不需要重置
#         "starting voice control": {"action": starting_voice_control, "reset": False}, # 不需要重置
#         "wagging your tail": {"action": action_wagging_your_tail, "reset": True},  # 需要重置
#         "stop action": {"action": stop_action, "reset": True},  # 需要重置
#     }

#     # 初始化关键词识别器
#     recognizer = KeywordRecognizer(keyword_actions)

#     # 开始监听
#     recognizer.start_listening()



# stage_num = None

# def action_hey_miro(stage_num, keyword_recognizer):
#     print("触发事件1：你好，Miro！")
#     keyword_recognizer.stop_listening()

#     stage_num = 1
#     print(f"Stage number updated to: {stage_num}")
#     # playback = AudioPlayback(interface, "/miro", '/home/yufeng/mdk/angry_dog_barking_mono_8000Hz.WAV')
#     # playback.play()




# def action_goodbye_miro(interface):
#     print("触发事件2：再见，Miro！")
#     playback = AudioPlayback(interface, "/miro", '/home/yufeng/mdk/output1.wav')
#     rospy.sleep(2)
#     playback.play()


# def starting_voice_control():
#     print("触发事件3：Starting voice control...")


# def action_wagging_your_tail():
#     print("触发事件4：坐下！摇尾巴！")


# def stop_action():
#     print("触发事件5：停止！")

# def show(interface):
#     print('show')
#     display = CameraDisplay(interface)
#     display.result_loop()

# def stop_show():
#     print('stop show')
#     cv2.destroyAllWindows()

