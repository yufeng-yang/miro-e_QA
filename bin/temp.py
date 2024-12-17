import numpy as np
# i = np.arange(0, 40, 2.5)
# print(i)
# x = np.interp(i, j, self.outbuf[:, c])
# '=================================================================='
# outbuf = np.array([
#     [0, 10, 20, 30],  # 样本 0: [LEFT, RIGHT, CENTRE, TAIL]
#     [1, 11, 21, 31],  # 样本 1: [LEFT, RIGHT, CENTRE, TAIL]
#     [2, 12, 22, 32],  # 样本 2: [LEFT, RIGHT, CENTRE, TAIL]
#     [3, 13, 23, 33],  # 样本 3: [LEFT, RIGHT, CENTRE, TAIL]
#     [4, 14, 24, 34],  # 样本 4: [LEFT, RIGHT, CENTRE, TAIL]
#     [5, 15, 25, 35],  # 样本 5: [LEFT, RIGHT, CENTRE, TAIL]
#     [6, 16, 26, 36],  # 样本 6: [LEFT, RIGHT, CENTRE, TAIL]
#     [7, 17, 27, 37],  # 样本 7: [LEFT, RIGHT, CENTRE, TAIL]
#     [8, 18, 28, 38],  # 样本 8: [LEFT, RIGHT, CENTRE, TAIL]
#     [9, 19, 29, 39],  # 样本 9: [LEFT, RIGHT, CENTRE, TAIL]
# ])

# playsamp = 3
# n_samp = 2
# spkrdata = outbuf[playsamp:(playsamp+n_samp),1]
# print(spkrdata)
# '=================================================================='
# import wave

# # 打开wav文件
# with wave.open('/home/yufeng/mdk/captured_audio.wav', 'rb') as wav_file:
#     # 获取通道数
#     n_channels = wav_file.getnchannels()
#     # 获取采样宽度（字节数）
#     sampwidth = wav_file.getsampwidth()
#     # 获取采样率
#     framerate = wav_file.getframerate()
#     # 获取帧数
#     n_frames = wav_file.getnframes()
#     # 计算音频时长
#     duration = n_frames / framerate

#     print(f"通道数: {n_channels}")
#     print(f"采样宽度（字节数）: {sampwidth}")
#     print(f"采样率: {framerate} Hz")
#     print(f"音频时长: {duration} 秒")

# import wave
# import numpy as np

# # 打开 WAV 文件
# with wave.open('/home/yufeng/mdk/bin/client_audio.wav', 'rb') as wav_file:
#     # 读取所有帧数据
#     audio_data = wav_file.readframes(wav_file.getnframes())
    
#     # 将音频数据转换为 numpy 数组，格式为 16 位小端序
#     audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
#     # 因为是双声道（2 channels），需要 reshape 成 (n_frames, 2)
#     audio_array = audio_array.reshape(-1, 4)

#     print('#########')
#     print(audio_array)


# import cv2

# def get_camera_max_resolution():
#     cap = cv2.VideoCapture(0)  # 打开摄像头，参数为摄像头索引，通常为 0
#     if not cap.isOpened():
#         print("无法打开摄像头")
#         return

#     # 遍历常见的分辨率
#     resolutions = [
#         (640, 480),  # VGA
#         (1280, 720), # HD
#         (1920, 1080), # Full HD
#         (2560, 1440), # QHD
#         (3840, 2160), # 4K UHD
#     ]

#     print("测试摄像头支持的分辨率：")
#     for width, height in resolutions:
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#         actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         if actual_width == width and actual_height == height:
#             print(f"支持分辨率: {width}x{height}")
#         else:
#             print(f"不支持分辨率: {width}x{height}，最大支持为 {actual_width}x{actual_height}")

#     cap.release()

# if __name__ == "__main__":
#     get_camera_max_resolution()


from modified_video import client  # 假设代码在另一个模块中
import rospy
# 初始化 ROS 节点
rospy.init_node("robot_camera_handler", anonymous=True)

# 使用 client 类
camera_client = client(["record"])  # 或 ["show"]
camera_client.loop()  # 开始录制或显示

#!/usr/bin/python3

# import rospy
# import cv2
# import threading
# import speech_recognition as sr
# from modified_video import client  # 从 client 模块引入
# from Listening_to_generate_respond import generate_single_respond
# from miro_e_full_tuned_model import separate_main
# from temp_audio_play import AudioPlayback
# import miro2 as miro

# def watching_and_listening():
#     """
#     录制视频和音频，同时处理语音和情绪识别
#     """
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     # 启动 ROS 节点和摄像头录制
#     rospy.init_node("watching_and_listening", anonymous=True)
#     camera_client = client(["record"])  # 使用左摄像头录制
#     threading.Thread(target=camera_client.loop).start()  # 启动摄像头录制线程

#     with microphone as source:
#         print("请开始说话，停止时将自动结束录制...")
#         recognizer.adjust_for_ambient_noise(source)

#         try:
#             # 监听音频，直到说话结束
#             audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)

#             # 保存音频文件
#             audio_path = "captured_audio.wav"
#             with open(audio_path, "wb") as f:
#                 f.write(audio.get_wav_data())
#             print(f"音频已保存到 {audio_path}")

#             # 停止摄像头录制
#             rospy.signal_shutdown("结束视频录制")
#             print("视频录制结束")

#             # 语音识别
#             text = recognizer.recognize_google(audio, language="en-US")

#             # 情绪识别
#             emtion_label, emotion_score = separate_main(
#                 '/tmp/client_video_left.avi', audio_path
#             )
#             print(f"(your mood is {emtion_label} with {emotion_score} confidence) you said: {text}")

#         except sr.UnknownValueError:
#             print("抱歉，无法理解你的话。")
#             emtion_label, text = "Unknown", ""
#         except sr.RequestError:
#             print("无法连接到语音识别服务。")
#             emtion_label, text = "Error", ""

#     return emtion_label, text

# def interactive_feedback(users_mood, user_input):
#     """
#     根据用户情绪和输入生成响应
#     """
#     if users_mood == 'Happy':
#         # 首先播放狗狗快乐的叫声
#         playback = AudioPlayback(interface, "/miro", '/home/yufeng/mdk/happy_dog_barking_mono_8000Hz.WAV')
#         rospy.sleep(2)
        
#         playback.play()
#         # 然后是回复
#         rospy.sleep(1)
#         sound_file = generate_single_respond(detected_emotion='cheerful', user_input=user_input)
#         playback = AudioPlayback(interface, "/miro", sound_file)
#         rospy.sleep(2)
#         playback.play()

#     elif users_mood == 'Sad':
#         playback = AudioPlayback(interface, "/miro", '/home/yufeng/mdk/sad_dog_barking_mono_8000Hz.WAV')
#         rospy.sleep(2)
#         playback.play()
#         # 然后是回复
#         rospy.sleep(1)
        
#     elif users_mood == 'Angry':
#         playback = AudioPlayback(interface, "/miro", '/home/yufeng/mdk/angry_dog_barking_mono_8000Hz.WAV')
#         rospy.sleep(2)
#         playback.play()
#         # 然后是回复
#         rospy.sleep(1)
        
#     else:
#         print("I'm sorry, I don't know how to respond to that.")

# if __name__ == "__main__":
#     # 初始化机器人接口
#     interface = miro.lib.RobotInterface() 

#     # 进行视频和语音采集以及情绪识别
#     emotion_label, user_input = watching_and_listening()

#     # 根据用户情绪和输入生成反馈
#     interactive_feedback(emotion_label, user_input)




