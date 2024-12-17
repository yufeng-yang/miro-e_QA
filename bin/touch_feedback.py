# import math
# import time
# import threading


# class SadMoodFeedback:
#     def __init__(self, interface):
#         self.interface = interface
#         self.is_tail_wagging = False
#         self.is_ear_flapping = False
#         self.tail_thread = None
#         self.ear_thread = None

#     def _tail_wagging_loop(self, amplitude=0.5, frequency=2):
#         while self.is_tail_wagging:
#             wag_angle = 0.5 + amplitude * 0.5 * math.sin(2 * math.pi * frequency * (time.time() % 1))
#             self.interface.set_cos(miro.constants.JOINT_WAG, wag_angle)
#             self.interface.set_cos(miro.constants.JOINT_DROOP, 0.5)
#             time.sleep(0.02)
#         # 恢复默认状态
#         self.interface.set_cos(miro.constants.JOINT_WAG, 0.5)
#         self.interface.set_cos(miro.constants.JOINT_DROOP, 0.5)

#     def _ear_flapping_loop(self, amplitude=0.5, frequency=1):
#         while self.is_ear_flapping:
#             ear_angle = 0.5 + amplitude * 0.5 * math.sin(2 * math.pi * frequency * (time.time() % 1))
#             self.interface.set_cos(miro.constants.JOINT_EAR_L, ear_angle)
#             self.interface.set_cos(miro.constants.JOINT_EAR_R, ear_angle)
#             time.sleep(0.02)
#         # 恢复默认状态
#         self.interface.set_cos(miro.constants.JOINT_EAR_L, 0.5)
#         self.interface.set_cos(miro.constants.JOINT_EAR_R, 0.5)

#     def start_tail_wagging(self):
#         if not self.is_tail_wagging:
#             print("Tail wagging started.")
#             self.is_tail_wagging = True
#             self.tail_thread = threading.Thread(target=self._tail_wagging_loop)
#             self.tail_thread.start()

#     def stop_tail_wagging(self):
#         if self.is_tail_wagging:
#             print("Tail wagging stopped.")
#             self.is_tail_wagging = False
#             if self.tail_thread:
#                 self.tail_thread.join()

#     def start_ear_flapping(self):
#         if not self.is_ear_flapping:
#             print("Ear flapping started.")
#             self.is_ear_flapping = True
#             self.ear_thread = threading.Thread(target=self._ear_flapping_loop)
#             self.ear_thread.start()

#     def stop_ear_flapping(self):
#         if self.is_ear_flapping:
#             print("Ear flapping stopped.")
#             self.is_ear_flapping = False
#             if self.ear_thread:
#                 self.ear_thread.join()

#     def touch_and_feedback(self):
#         b = self.interface.get_touch_body()
#         print("Touch sensor output:", b)

#         if any(b):  # 如果数组中有 True，表示触摸事件
#             if not self.is_tail_wagging:
#                 self.start_tail_wagging()
#             if not self.is_ear_flapping:
#                 self.start_ear_flapping()
#         else:  # 如果没有触摸事件
#             self.stop_tail_wagging()
#             self.stop_ear_flapping()


# if __name__ == "__main__":
#     import rospy
#     import miro2 as miro

#     # 假设已经初始化了 interface 对象
#     interface = miro.lib.RobotInterface()

#     # 创建控制器对象
#     controller = SadMoodFeedback(interface)

#     # 主循环
#     while not rospy.is_shutdown():
#         controller.touch_and_feedback()
#         rospy.sleep(0.1)  # 添加适当的延迟，避免过于频繁地轮询





import math
import time
import threading
import rospy
import miro2 as miro
import numpy as np


class SadMoodFeedback:
    def __init__(self, interface):
        self.interface = interface

        # 状态变量
        self.is_tail_wagging = False
        self.is_ear_flapping = False

        # 机器人位置
        self.pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.total_angle = 0.0  # 累积旋转角度（弧度）
        self.last_angle = 0.0  # 上一次记录的角度（弧度）

    def start_tail_wagging(self, amplitude=0.5, frequency=2):
        """尾巴摇动动作"""
        if not self.is_tail_wagging:
            self.is_tail_wagging = True
            threading.Thread(target=self._tail_wagging_loop, args=(amplitude, frequency)).start()

    def stop_tail_wagging(self):
        """停止尾巴摇动"""
        self.is_tail_wagging = False
        self.interface.set_cos(miro.constants.JOINT_WAG, 0.5)

    def _tail_wagging_loop(self, amplitude, frequency):
        """尾巴摇动逻辑"""
        while self.is_tail_wagging:
            wag_angle = 0.5 + amplitude * 0.5 * math.sin(2 * math.pi * frequency * time.time())
            self.interface.set_cos(miro.constants.JOINT_WAG, wag_angle)
            time.sleep(0.02)

    def start_ear_flapping(self, amplitude=0.5, frequency=1):
        """耳朵摆动动作"""
        if not self.is_ear_flapping:
            self.is_ear_flapping = True
            threading.Thread(target=self._ear_flapping_loop, args=(amplitude, frequency)).start()

    def stop_ear_flapping(self):
        """停止耳朵摆动"""
        self.is_ear_flapping = False
        self.interface.set_cos(miro.constants.JOINT_EAR_L, 0.5)
        self.interface.set_cos(miro.constants.JOINT_EAR_R, 0.5)

    def _ear_flapping_loop(self, amplitude, frequency):
        """耳朵摆动逻辑"""
        while self.is_ear_flapping:
            ear_angle = 0.5 + amplitude * math.sin(2 * math.pi * frequency * time.time())
            self.interface.set_cos(miro.constants.JOINT_EAR_L, ear_angle)
            self.interface.set_cos(miro.constants.JOINT_EAR_R, ear_angle)
            time.sleep(0.02)

    def touch_and_feedback(self):
        """检测触摸并触发反馈"""
        while not rospy.is_shutdown():
            b = self.interface.get_touch_body()
            print("Touch sensor output:", b)

            if any(b):  # 如果检测到触摸
                if not self.is_tail_wagging:
                    self.start_tail_wagging()
                if not self.is_ear_flapping:
                    self.start_ear_flapping()
            else:  # 如果未检测到触摸
                self.stop_tail_wagging()
                self.stop_ear_flapping()
            time.sleep(0.1)  # 控制检测频率

    def spin(self, lin_vel=0.02, ang_vel=0.785, duration=8):
        """旋转功能"""
        for _ in range(int(duration / 0.1)):
            self.interface.set_vel(lin_vel=lin_vel, ang_vel=ang_vel)
            time.sleep(0.1)
        self.interface.set_vel(lin_vel=0.0, ang_vel=0.0)

    def light_function(self):
        """灯光控制"""
        colors = [(0, 0, 64), (0, 0, 128), (0, 0, 192), (0, 0, 255)]
        for color in colors:
            self.interface.set_illum(0, color, 100)
            time.sleep(0.5)

    def execute_all_functions(self, duration=5):
        """多线程执行所有功能"""
        light_thread = threading.Thread(target=self.light_function)
        touch_thread = threading.Thread(target=self.touch_and_feedback)

        # 启动线程
        light_thread.start()
        touch_thread.start()

        # 等待线程完成
        light_thread.join()
        touch_thread.join()


if __name__ == "__main__":
    interface = miro.lib.RobotInterface()
    controller = SadMoodFeedback(interface)

    # 多线程执行功能
    threading.Thread(target=controller.execute_all_functions).start()

    rospy.spin()  # 运行 ROS 主循环
