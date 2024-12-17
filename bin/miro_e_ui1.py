import tkinter as tk
from PIL import Image, ImageTk
import sounddevice as sd
from scipy.io.wavfile import write
import cv2
import threading
import numpy as np
import time

from miro_e_full_tuned_model import separate_main
from extract_video_feature_tools import initialize_face_detector, load_emotion_model

# 识别部分的导包
from au_voice_to_text import AudioFileRecognizer

# 机器人控制方面的导包
from temp_audio_play import AudioPlayback
import miro2 as miro
import rospy
# 快乐的回复即整体动作
from miro_e_function_test import HappyMoodFeedback, execute_all_functions
# 悲伤的导包
from miro_e_sad_show import SadMoodFeedback, execute_all_sad_functions

# 回复功能的导包
from Listening_to_generate_respond import generate_single_respond

class ui_preparation:
    def __init__(self, root, title="机器人界面", width=800, height=600, mediapipe_model=None, emotion_model=None, robot_interface=None, robot_status=None):
        self.root = root
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        
        self.top_text = "None"
        self.status_label = tk.Label(self.root, text=self.top_text, font=("Arial", 16), bg="lightgrey", anchor="w")
        self.status_label.pack(fill=tk.X, pady=10)
        
        self.frame_left = tk.Frame(self.root)
        self.frame_left.pack(side=tk.LEFT, padx=50, pady=100)
        
        self.frame_right = tk.Frame(self.root)
        self.frame_right.pack(side=tk.RIGHT, padx=100, pady=10)
        
        self.fs = 44100
        self.recording = False
        self.audio_data = []
        self.video_writer = None
        self.load_images_and_create_buttons()
        # 事件控制
        self.status = None

        # 一些模型的输入
        self.mediapipe_model = mediapipe_model
        self.emotion_model = emotion_model

        self.robot_interface = robot_interface


        # 用于标注机器人状态的变量
        self.robot_status = robot_status
    
    def resize_image(self, image_path, width, height):
        image = Image.open(image_path)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)
    
    def update_status(self, new_status):
        self.status_label.config(text=new_status)
    
    def start_recording(self, event=None):
        self.update_status("开始录音和录像...")
        self.recording = True
        self.audio_data = []
        self.start_time = time.time()
        
        self.record_thread = threading.Thread(target=self.record_audio_and_video)
        self.record_thread.start()

    # def stop_recording(self, event=None):
    #     self.update_status("停止录音和录像，保存文件中...")
    #     self.recording = False
    #     self.record_thread.join()
        
    #     # 保存音频数据
    #     audio_array = np.array(self.audio_data, dtype='int16')
    #     audio_path = "/home/yufeng/mdk/user_input.wav"
    #     write(audio_path, self.fs, audio_array)
        
    #     # 停止并保存视频
    #     if self.video_writer is not None:
    #         self.video_writer.release()
    #         self.video_writer = None
    #     video_path = "/home/yufeng/mdk/user_video.avi"
        
    #     self.update_status("I am Thinking...")
    #     self.status = "Thinking"
        
    #     # 在后台线程中执行融合逻辑
    #     def process_logic():
    #         try:
    #             a, b = separate_main(video_path, audio_path, self.mediapipe_model, self.emotion_model)
    #             self.update_status(f"{a} with {b:.2f}")
    #         except Exception as e:
    #             self.update_status(f"处理错误：{e}")
    #             print(f"Error during processing: {e}")
        
    #     # 启动线程
    #     process_thread = threading.Thread(target=process_logic)
    #     process_thread.start()
    
    def stop_recording(self, event=None):
        self.update_status("停止录音和录像，保存文件中...")
        self.recording = False
        self.record_thread.join()
        
        # 保存音频数据
        audio_array = np.array(self.audio_data, dtype='int16')
        audio_path = "/home/yufeng/mdk/user_input.wav"
        write(audio_path, self.fs, audio_array)
        
        # 停止并保存视频
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        video_path = "/home/yufeng/mdk/user_video.avi"
        
        self.update_status("I am Thinking...")
        self.status = "Thinking"
        
        # 在后台线程中执行融合逻辑
        def process_logic():
            try:
                # 创建音频识别器并提取文本
                audio_recognizer = AudioFileRecognizer(audio_path)
                text = audio_recognizer.recognize_from_audio_file()
                
                if text:
                    print(f"识别的文本内容：{text}")
                else:
                    print("未能识别出文本。")
                    text = None
                
                # 调用融合逻辑
                a, b = separate_main(video_path, audio_path, self.mediapipe_model, self.emotion_model)
                self.update_status(f"{a} with {b:.2f} and {text}")
                print(f"you said: {text}")

                # 现在有了文本和emotion，可以进行回复
                # 1. 通过文本识别，判断是否有关键词
                if 'goodbye' in text.lower():
                    playback = AudioPlayback(self.robot_interface, "/miro", '/home/yufeng/mdk/good_bye.wav')
                    rospy.sleep(1)
                    playback.play()
                
                # 2. 没有关键字，进入情感部分
                else:
                    # 2.1 情感部分
                    if a == 'Sad':
                        playback = AudioPlayback(self.robot_interface, "/miro", '/home/yufeng/mdk/sad_dog_barking_mono_8000Hz.WAV')
                        playback.play()
                        # 播放悲伤的预设回复
                        controller = SadMoodFeedback(interface,a)
                        execute_all_sad_functions(controller, duration=9)

                    
                    elif a == 'Angry':
                        playback = AudioPlayback(self.robot_interface, "/miro", '/home/yufeng/mdk/angry_dog_barking_mono_8000Hz.WAV')
                        playback.play()
                        rospy.sleep(1)

                    
                    elif a == 'Happy':
                        playback = AudioPlayback(self.robot_interface, "/miro", '/home/yufeng/mdk/happy_dog_barking_mono_8000Hz.WAV')
                        playback.play()
                        sound_file = generate_single_respond(detected_emotion='cheerful', user_input=text)
                        if sound_file is not None:
                            controller = HappyMoodFeedback(interface)
                            execute_all_functions(controller, duration=9)
                    
                    else:
                        sound_file = generate_single_respond(detected_emotion='happy', user_input=text)
                        if sound_file is not None:
                            playback = AudioPlayback(self.robot_interface, "/miro", sound_file)
                            rospy.sleep(1)
                            playback.play()



            except Exception as e:
                self.update_status(f"处理错误：{e}")
                print(f"Error during processing: {e}")
        
        # 启动线程
        process_thread = threading.Thread(target=process_logic)
        process_thread.start()

    def record_audio_and_video(self):
        try:
            cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter("user_video.avi", fourcc, 10, (640, 480))
            
            while self.recording:
                # 音频录制
                audio_chunk = sd.rec(int(self.fs * 0.1), samplerate=self.fs, channels=1, dtype='int16')
                sd.wait()
                self.audio_data.extend(audio_chunk.flatten())
                
                # 视频录制
                ret, frame = cap.read()
                if ret:
                    self.video_writer.write(frame)
            
            cap.release()
        
        except Exception as e:
            self.update_status(f"录制错误：{e}")
    
    def load_images_and_create_buttons(self):
        try:
            image1 = self.resize_image("/home/yufeng/mdk/bin/ui_listen_and_respond.png", 300, 300)
            image2 = self.resize_image("/home/yufeng/mdk/bin/Q2.png", 150, 150)
            image3 = self.resize_image("/home/yufeng/mdk/bin/Q2.png", 150, 150)
            
            button1 = tk.Button(self.frame_left, image=image1)
            button1.image = image1
            button1.pack()
            
            button1.bind("<ButtonPress-1>", self.start_recording)
            button1.bind("<ButtonRelease-1>", self.stop_recording)
            
            button2 = tk.Button(self.frame_right, image=image2, command=lambda: self.button_action(2))
            button2.image = image2
            button2.pack(pady=10)
            
            button3 = tk.Button(self.frame_right, image=image3, command=lambda: self.button_action(3))
            button3.image = image3
            button3.pack()
        
        except Exception as e:
            self.update_status(f"错误：无法加载图片。{e}")

# 主程序入口
if __name__ == "__main__":
    # 初始化模型
    mediapipe_model = initialize_face_detector()
    emotion_model = load_emotion_model()

    # 初始化机器人
    interface = miro.lib.RobotInterface()

    print("模型初始化完成")

    # ui界面的展示
    root = tk.Tk()
    app = ui_preparation(root, mediapipe_model=mediapipe_model, emotion_model=emotion_model, robot_interface=interface)
    root.mainloop()
