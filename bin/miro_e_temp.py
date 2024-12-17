from miro_e_keyword_trigger import KeywordRecognizer
from temp_audio_play import AudioPlayback
import rospy
import miro2 as miro
# 配置关键词、触发方法和是否需要重置
from functools import partial

import cv2

# 关于ui界面的import
import tkinter as tk
from miro_e_ui1 import ui_preparation

# 载入展示界面
from miro_e_mediapipe_add_DDAMDN import CameraDisplay

# 准确模型
from extract_video_feature_tools import initialize_face_detector, load_emotion_model

# 最开始，初始化机器人接口，这个自动初始化nodes了
interface = miro.lib.RobotInterface()

print("机器人接口初始化完成")

# 初始化模型
mediapipe_model = initialize_face_detector()
emotion_model = load_emotion_model()

print("模型初始化完成")

# ui界面的展示
root = tk.Tk()
app = ui_preparation(root, mediapipe_model=mediapipe_model, emotion_model=emotion_model)
root.mainloop()



