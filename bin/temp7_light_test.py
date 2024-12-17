# import rospy
# from std_msgs.msg import UInt16MultiArray, Int16MultiArray

# import time
# import sys
# import os
# import numpy as np
# import wave, struct

# import miro2 as miro

# interface = miro.lib.RobotInterface()
# # interface.post_tone(freq=1000, dur=500, vol=128)  # 播放 1000Hz 的音调，持续 0.5 秒
# # 定义红色到黄色之间的渐变颜色
# colors = [
#     (255, 0, 0),   # 红色
#     (255, 64, 0),  # 橙红色
#     (255, 128, 0), # 橙色
#     (255, 192, 0), # 橙黄色
#     (255, 255, 0)  # 黄色
# ]

# # 让灯光以前中后组交替闪烁 5 秒钟
# for i in range(25):
#     color_index = i % len(colors)
#     current_color = colors[color_index]

#     # 每次逐渐减少亮度
#     for brightness in range(200, 0, -40):  # 从 200 逐渐减到 0，步长为 40
#         if i % 3 == 0:
#             # 前部 LED 闪烁
#             interface.set_illum(miro.constants.ILLUM_FRONT, current_color, brightness)
#         elif i % 3 == 1:
#             # 中部 LED 闪烁
#             interface.set_illum(miro.constants.ILLUM_MID, current_color, brightness)
#         else:
#             # 后部 LED 闪烁
#             interface.set_illum(miro.constants.ILLUM_REAR, current_color, brightness)

#         # 每 0.05 秒逐渐降低亮度
#         time.sleep(0.05)

#     # 每 0.2 秒切换一次 LED 组
#     time.sleep(0.15)

# # 最后将所有灯光设置为稳定的黄色，亮度恢复到100
# interface.set_illum(miro.constants.ILLUM_ALL, (255, 255, 0), 100)


import math
import rospy
from std_msgs.msg import UInt16MultiArray, Int16MultiArray

import time
import sys
import os
import numpy as np
import wave, struct
import threading

import miro2 as miro
from temp_audio_play import AudioPlayback

import random


class GoodWeatherFeedback:
    def __init__(self, interface, mood):
        self.interface = interface
        self.mood = mood

    def light_function(self):
        if self.mood == "happy":
            colors = [
                (255, 0, 0),   # 红色
                (255, 64, 0),  # 橙红色
                (255, 128, 0), # 橙色
                (255, 192, 0), # 橙黄色
                (255, 255, 0)  # 黄色
            ]

            # 让灯光以前中后组交替闪烁
            for i in range(25):
                color_index = i % len(colors)
                current_color = colors[color_index]

                # 每次逐渐减少亮度
                for brightness in range(200, 0, -40):  # 从 200 逐渐减到 0，步长为 40
                    if i % 3 == 0:
                        # 前部 LED 闪烁
                        self.interface.set_illum(miro.constants.ILLUM_FRONT, current_color, brightness)
                    elif i % 3 == 1:
                        # 中部 LED 闪烁
                        self.interface.set_illum(miro.constants.ILLUM_MID, current_color, brightness)
                    else:
                        # 后部 LED 闪烁
                        self.interface.set_illum(miro.constants.ILLUM_REAR, current_color, brightness)

                    # 每 0.05 秒逐渐降低亮度
                    time.sleep(0.05)

                # 每 0.2 秒切换一次 LED 组
                time.sleep(0.15)

            # 最后将所有灯光设置为稳定的黄色，亮度恢复到100
            self.interface.set_illum(miro.constants.ILLUM_ALL, (255, 255, 0), 100)

    def play_music(self):
        if self.mood == "happy":
            # play = AudioPlayback(self.interface, "/miro", "/home/yufeng/mdk/bin/good_weather_feedback.wav")
            play = AudioPlayback(self.interface, "/miro", "/home/yufeng/mdk/bin/Question1_mono.wav")
            play.play()
    
    def walk_around(self):
        if self.mood == "happy":
            # 设置线速度和角速度来让机器人做圆周运动
            lin_vel = 0.3  # 线速度 (正值表示向前移动)
            ang_vel = 1.0  # 角速度 (正值表示顺时针，负值表示逆时针)
            motion_time  = 12

            for _ in range(motion_time*10):
                self.interface.set_vel(lin_vel, ang_vel)
                time.sleep(0.1)

            # 停止运动
            self.interface.set_vel(lin_vel=0.0, ang_vel=0.0)
            

        # if self.mood == "sad":
        #     # 设置线速度和角速度来让机器人做圆周运动
        #     # lin_vel = 0.3  # 线速度 (正值表示向前移动)
        #     ang_vel = 0.5  # 角速度 (正值表示顺时针，负值表示逆时针)
        #     motion_time  = 10

        #     for _ in range(motion_time*10):
        #         self.interface.set_vel(lin_vel, ang_vel)
        #         time.sleep(0.1)

        #     # 停止运动
        #     self.interface.set_vel(lin_vel=0.0, ang_vel=0.0)


        
    def wag_tail(self):
        if self.mood == "happy":
            # 设置尾巴上下左右摆动的持续时间
            tail_motion_time = 12  # 总运动时间（秒）
            start_time = time.time()

            # 定义上下左右的摇摆范围
            wag_min = 0.0  # 左右摇摆的最小值
            wag_max = 1.0  # 左右摇摆的最大值
            droop_min = 0.0  # 上下摆动的最小值（尾巴上）
            droop_max = 0.5  # 上下摆动的最大值（尾巴下）

            while time.time() - start_time < tail_motion_time:
                for i in range(20):  # 控制频率，循环次数可以调整
                    # 让尾巴左右摇摆
                    wag_value = wag_min + (wag_max - wag_min) * ((i % 2) == 0)  # 左右摆动
                    self.interface.set_cos(miro.constants.JOINT_WAG, wag_value)
                    self.interface.set_cos(miro.constants.JOINT_DROOP, 0)

                    # # 让尾巴上下摆动
                    # droop_value = droop_min + (droop_max - droop_min) * ((i % 2) == 0)  # 上下摆动
                    # self.interface.set_cos(miro.constants.JOINT_DROOP, droop_value)

                    # 每次摇摆后暂停 0.1 秒
                    time.sleep(0.1)

            # 停止尾巴运动，恢复默认角度
            self.interface.set_cos(miro.constants.JOINT_WAG, miro.constants.WAG_CALIB)  # 恢复默认摆动角度
            self.interface.set_cos(miro.constants.JOINT_DROOP, miro.constants.DROOP_CALIB)  # 恢复默认垂直角度

        if self.mood == "sad":
            # 设置尾巴上下左右摆动的持续时间
            tail_motion_time = 10  # 总运动时间（秒）
            start_time = time.time()

            # 定义上下左右的摇摆范围
            wag_min = 0.0  # 左右摇摆的最小值
            wag_max = 1  # 左右摇摆的最大值
            droop_min = 0.0  # 上下摆动的最小值（尾巴上）
            droop_max = 0.5  # 上下摆动的最大值（尾巴下）

            while time.time() - start_time < tail_motion_time:
                for i in range(20):  # 控制频率，循环次数可以调整
                    # 让尾巴左右摇摆
                    wag_value = wag_min + (wag_max - wag_min) * ((i % 2) == 0)  # 左右摆动
                    self.interface.set_cos(miro.constants.JOINT_WAG, wag_value)
                    self.interface.set_cos(miro.constants.JOINT_DROOP, 0.5)

                    # # 让尾巴上下摆动
                    # droop_value = droop_min + (droop_max - droop_min) * ((i % 2) == 0)  # 上下摆动
                    # self.interface.set_cos(miro.constants.JOINT_DROOP, droop_value)

                    # 每次摇摆后暂停 0.1 秒
                    time.sleep(0.5)

            # 停止尾巴运动，恢复默认角度
            self.interface.set_cos(miro.constants.JOINT_WAG, miro.constants.WAG_CALIB)  # 恢复默认摆动角度
            self.interface.set_cos(miro.constants.JOINT_DROOP, miro.constants.DROOP_CALIB)  # 恢复默认垂直角度

    def head_control(self):
        if self.mood == "happy":
            # # 抬头到上半部分
            # lift_up_angle = (miro.constants.LIFT_RAD_MIN + miro.constants.LIFT_RAD_MAX) / 2  # 上半部分的起点
            # self.interface.set_kin(miro.constants.JOINT_LIFT, lift_up_angle)
            # print(lift_up_angle)

            # 设置运动持续时间为 12 秒
            motion_time = 12
            start_time = time.time()
            # # 随机生成 45 到 60 度之间的角度，并转换为弧度
            # random_lift_angle_deg = random.randint(45,60)
            # random_lift_angle_rad = math.radians(random_lift_angle_deg)
            # # 设置 Lift 角度为随机生成的角度
            # self.interface.set_kin(miro.constants.JOINT_LIFT, random_lift_angle_rad)
            # print(random_lift_angle_rad)

            # 这个俯仰角好像是倒着的。越小越大，并且是弧度制单位
            self.interface.set_kin(miro.constants.JOINT_LIFT, miro.constants.LIFT_RAD_CALIB-0.3)
            # print(miro.constants.LIFT_RAD_CALIB)

            # 持续摇动头部 12 秒
            while time.time() - start_time < motion_time:

                # 向右摇头
                self.interface.set_kin(miro.constants.JOINT_YAW, miro.constants.YAW_RAD_MAX / 2)
                time.sleep(0.5)

                # 向左摇头
                self.interface.set_kin(miro.constants.JOINT_YAW, miro.constants.YAW_RAD_MIN / 2)
                time.sleep(0.5)

                self.interface.set_kin(miro.constants.JOINT_PITCH, miro.constants.PITCH_RAD_MAX)
                time.sleep(0.5)

                self.interface.set_kin(miro.constants.JOINT_PITCH, 0)
                time.sleep(0.5)
                # # 向上抬头（向上倾斜）
                # self.interface.set_kin(miro.constants.JOINT_PITCH, miro.constants.PITCH_RAD_MAX / 2)
                # time.sleep(0.5)

                # # 向下低头（向下倾斜）
                # self.interface.set_kin(miro.constants.JOINT_PITCH, miro.constants.PITCH_RAD_MIN / 2)
                # time.sleep(0.5)

            # 停止运动并恢复到默认位置
            self.interface.set_kin(miro.constants.JOINT_LIFT, miro.constants.LIFT_RAD_CALIB)
            self.interface.set_kin(miro.constants.JOINT_YAW, miro.constants.YAW_RAD_CALIB)
            self.interface.set_kin(miro.constants.JOINT_PITCH, miro.constants.PITCH_RAD_CALIB)

        if self.mood == "sad":
            # # 抬头到上半部分
            # lift_up_angle = (miro.constants.LIFT_RAD_MIN + miro.constants.LIFT_RAD_MAX) / 2  # 上半部分的起点
            # self.interface.set_kin(miro.constants.JOINT_LIFT, lift_up_angle)
            # print(lift_up_angle)

            # 设置运动持续时间为 12 秒
            motion_time = 10
            start_time = time.time()
            # # 随机生成 45 到 60 度之间的角度，并转换为弧度
            # random_lift_angle_deg = random.randint(45,60)
            # random_lift_angle_rad = math.radians(random_lift_angle_deg)
            # # 设置 Lift 角度为随机生成的角度
            # self.interface.set_kin(miro.constants.JOINT_LIFT, random_lift_angle_rad)
            # print(random_lift_angle_rad)

            # 这个俯仰角好像是倒着的。越小越大，并且是弧度制单位
            self.interface.set_kin(miro.constants.JOINT_LIFT, miro.constants.LIFT_RAD_CALIB+0.4)
            # print(miro.constants.LIFT_RAD_CALIB)

            # 持续摇动头部 12 秒
            while time.time() - start_time < motion_time:

                # 向右摇头
                self.interface.set_kin(miro.constants.JOINT_YAW, miro.constants.YAW_RAD_MAX / 2)
                time.sleep(0.5)

                # 向左摇头
                self.interface.set_kin(miro.constants.JOINT_YAW, miro.constants.YAW_RAD_MIN / 2)
                time.sleep(0.5)

                self.interface.set_kin(miro.constants.JOINT_PITCH, miro.constants.PITCH_RAD_MAX)
                time.sleep(0.5)

                self.interface.set_kin(miro.constants.JOINT_PITCH, 0)
                time.sleep(0.5)
                # # 向上抬头（向上倾斜）
                # self.interface.set_kin(miro.constants.JOINT_PITCH, miro.constants.PITCH_RAD_MAX / 2)
                # time.sleep(0.5)

                # # 向下低头（向下倾斜）
                # self.interface.set_kin(miro.constants.JOINT_PITCH, miro.constants.PITCH_RAD_MIN / 2)
                # time.sleep(0.5)

            # 停止运动并恢复到默认位置
            self.interface.set_kin(miro.constants.JOINT_LIFT, miro.constants.LIFT_RAD_CALIB)
            self.interface.set_kin(miro.constants.JOINT_YAW, miro.constants.YAW_RAD_CALIB)
            self.interface.set_kin(miro.constants.JOINT_PITCH, miro.constants.PITCH_RAD_CALIB)



    def main(self):
        # 创建线程来并行处理灯光和音频播放
        light_thread = threading.Thread(target=self.light_function)
        # music_thread = threading.Thread(target=self.play_music)
        motion_thread = threading.Thread(target=self.walk_around)
        tail_thread = threading.Thread(target=self.wag_tail)
        head_thread = threading.Thread(target=self.head_control)

        # 启动线程
        light_thread.start()
        # music_thread.start()
        motion_thread.start()
        tail_thread.start()
        head_thread.start()

        # 等待线程完成
        light_thread.join()
        # music_thread.join()
        motion_thread.join()
        tail_thread.join()
        head_thread.join()

# 示例使用
if __name__ == "__main__":
    # rospy.init_node("good_weather_feedback_node")

    # 假设已经初始化了 interface 对象
    interface = miro.lib.RobotInterface()
    mood = "happy"

    feedback = GoodWeatherFeedback(interface, mood)
    feedback.main()
