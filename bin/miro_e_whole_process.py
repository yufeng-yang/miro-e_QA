import speech_recognition as sr
import cv2
import threading

from miro_e_full_tuned_model import main, separate_main
from Listening_to_generate_respond import generate_single_respond


from temp_audio_play import AudioPlayback

import miro2 as miro
import rospy

# from modified_video import client
from miro_e_function_test import HappyMoodFeedback, execute_all_functions

class VideoRecorder:
    def __init__(self, output_path):
        self.output_path = output_path
        self.cap = cv2.VideoCapture(0)  # 打开摄像头
        
        # 设置摄像头分辨率为 640x480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 确保分辨率设置正确
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width != 640 or actual_height != 480:
            print(f"警告：摄像头无法设置为 640x480，实际分辨率为 {actual_width}x{actual_height}")
        else:
            print("摄像头成功设置为 640x480")

        # 初始化视频写入对象，编码器不变
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, 20.0, (640, 480))  # 设置输出分辨率
        self.recording = True  # 用于控制录制状态


    def start(self):
        """
        开始录制视频
        """
        self.thread = threading.Thread(target=self.record)
        self.thread.start()

    def record(self):
        """
        实际的录制逻辑
        """
        while self.recording:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)
            else:
                print("无法从摄像头读取帧。")
                break

    def stop(self):
        """
        停止录制视频
        """
        self.recording = False
        self.thread.join()  # 等待线程完成
        self.cap.release()
        self.out.release()
        print(f"视频已保存到 {self.output_path}")

def watching_and_listening():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    video_recorder = VideoRecorder("captured_video.mp4")

    with microphone as source:
        print("请开始说话，停止时将自动结束录制...")
        recognizer.adjust_for_ambient_noise(source)
        # 启动视频录制
        video_recorder.start()

        try:
            # 监听音频，直到说话结束
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)

            # 保存音频文件
            audio_path = "captured_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())
            print(f"音频已保存到 {audio_path}")

            # 识别语音
            print("recognizing and judging your mood...")
            # 停止视频录制
            video_recorder.stop()
            # 语音识别
            text = recognizer.recognize_google(audio, language="en-US")
            # 情绪识别
            emtion_label,emotion_score = separate_main('/home/yufeng/mdk/captured_video.mp4','/home/yufeng/mdk/captured_audio.wav')
            print(f"(your mood is {emtion_label} with {emotion_score} confidence) you said: {text}")
            
        except sr.UnknownValueError:
            print("抱歉，无法理解你的话。")
        except sr.RequestError:
            print("无法连接到语音识别服务。")

    
    return emtion_label, text

# =========================================================================================================================================================
# import speech_recognition as sr
# import threading
# from miro_e_mediapipe_add_DDAMDN import CameraDisplay
# from extract_video_feature_tools import feature_to_7probability


# class VideoRecorder:
#     def __init__(self, output_path):
#         self.output_path = output_path
#         self.recording = True

#     def start(self):
#         """
#         开始情绪识别（替代原本的视频录制逻辑）
#         """
#         self.thread = threading.Thread(target=self.record)
#         self.thread.start()

#     def record(self):
#         """
#         实时情绪识别逻辑
#         """
#         self.display = CameraDisplay()
#         self.display.result_loop()

#     def stop(self):
#         """
#         停止情绪识别并获取累积特征
#         """
#         rospy.signal_shutdown('结束情绪识别')
#         self.thread.join()
#         print("情绪识别已停止")
#         return self.get_average_feature()

#     def get_average_feature(self):
#         """
#         获取累积的特征并返回
#         """
#         return self.display.get_accumulated_feature()


# def watching_and_listening():
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     # 初始化情绪识别
#     video_recorder = VideoRecorder("captured_video.mp4")

#     with microphone as source:
#         print("请开始说话，停止时将自动结束录制...")
#         recognizer.adjust_for_ambient_noise(source)
#         # 启动情绪识别
#         video_recorder.start()

#         try:
#             # 监听音频
#             audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)

#             # 停止情绪识别
#             average_feature = video_recorder.stop()

#             # 保存音频文件
#             audio_path = "captured_audio.wav"
#             with open(audio_path, "wb") as f:
#                 f.write(audio.get_wav_data())
#             print(f"音频已保存到 {audio_path}")

#             # 语音识别
#             print("Recognizing speech...")
#             text = recognizer.recognize_google(audio, language="en-US")
#             print(f"You said: {text}")

#             # 情绪识别结果
#             if average_feature is not None:
#                 video_probabilities = feature_to_7probability(None, average_feature)
#                 emotion_label = max(video_probabilities, key=video_probabilities.get)
#                 emotion_score = video_probabilities[emotion_label]
#                 print(f"Detected emotion: {emotion_label} with confidence {emotion_score}")
#             else:
#                 emotion_label = None
#                 emotion_score = None
#                 print("No emotion detected.")

#         except sr.UnknownValueError:
#             print("抱歉，无法理解你的话。")
#             text = None
#             emotion_label = None
#             emotion_score = None
#         except sr.RequestError:
#             print("无法连接到语音识别服务。")
#             text = None
#             emotion_label = None
#             emotion_score = None

#     # 返回语音和情绪识别结果
#     return emotion_label, emotion_score, text


# # # 使用示例
# # if __name__ == "__main__":
# #     emotion_label, emotion_score, text = watching_and_listening()
# #     print(f"Text: {text}")
# #     print(f"Emotion Label: {emotion_label}")
# #     print(f"Emotion Score: {emotion_score}")

# # =========================================================================================================================================================


# # 会卡住原因未知，所以放到主函数中，直接做判断了
# # def interactive_feedback(users_mood, user_input, interface):
# #     if users_mood == 'Happy':
# #         # 首先播放狗狗快乐的叫声
# #         playback = AudioPlayback(interface, "/miro", '/home/yufeng/mdk/happy_dog_barking_mono_8000Hz.WAV')
# #         rospy.sleep(2)
# #         playback.play()
# #         # 然后是回复
# #         # 创建控制器对象
# #         controller = HappyMoodFeedback(interface)
# #         while not rospy.is_shutdown():
# #             sound_file = generate_single_respond(detected_emotion='cheerful', user_input=user_input)
# #             if sound_file is not None:
# #                 # controller.happy_audio(sound_file)
# #                 execute_all_functions(controller, duration=9)
# #         rospy.sleep(1)

# #     elif users_mood == 'Sad':
# #         playback = AudioPlayback(interface, "/miro", '/home/yufeng/mdk/sad_dog_barking_mono_8000Hz.WAV')
# #         rospy.sleep(2)
# #         playback.play()
# #         # 然后是回复
# #         rospy.sleep(1)
        
# #     elif users_mood == 'Angry':
# #         playback = AudioPlayback(interface, "/miro", '/home/yufeng/mdk/angry_dog_barking_mono_8000Hz.WAV')
# #         rospy.sleep(2)
# #         playback.play()
# #         # 然后是回复
# #         rospy.sleep(1)
        
    
# #     else:
# #         playback = AudioPlayback(interface, "/miro", '/home/yufeng/mdk/sad_dog_barking_mono_8000Hz.WAV')

if __name__ == "__main__":
    # interface = miro.lib.RobotInterface() 
    interface = None
    emotion_label, text = watching_and_listening()
    emotion_label = 'Happy'
    text = 'How are you?'
    print(emotion_label, text)
    if emotion_label == 'Happy':
        sound_file = '/home/yufeng/mdk/happy_dog_barking_mono_8000Hz.WAV'
        playback = AudioPlayback(interface, "/miro", sound_file)
        rospy.sleep(2)
        playback.play()
        controller = HappyMoodFeedback(interface)
        while not rospy.is_shutdown():
            sound_file = generate_single_respond(detected_emotion='cheerful', user_input=text)
            if sound_file is not None:
                # controller.happy_audio(sound_file)
                execute_all_functions(controller, duration=9)
                emotion_label, text = watching_and_listening()
                # emotion_label = 'Happy'
                
    
    elif emotion_label == 'Sad':
        sound_file = '/home/yufeng/mdk/sad_dog_barking_mono_8000Hz.WAV'
        playback = AudioPlayback(interface, "/miro", sound_file)
        rospy.sleep(2)
        playback.play()

    
    elif emotion_label == 'Angry':
        sound_file = '/home/yufeng/mdk/angry_dog_barking_mono_8000Hz.WAV'
        playback = AudioPlayback(interface, "/miro", sound_file)
        rospy.sleep(2)
        playback.play()
    
    else:
        print('1111')








