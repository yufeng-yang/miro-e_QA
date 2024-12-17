#################################################################
# 基于时间的运动控制，被放弃，因为我发现存在pose的控制
# import time
# import miro2 as miro
# class happy_mood_motion():
#     def __init__(self, interface):
#         self.interface = interface

#     def spin(self, duration=8):

#             # 设置线速度和角速度
#             lin_vel = 0.01  # 设置较低线速度，表示小幅前进
#             ang_vel = 0.785 # 设置角速度，表示旋转

#             for _ in range(int(duration / 0.1)):
#                 self.interface.set_vel(lin_vel=lin_vel, ang_vel=ang_vel)
#                 time.sleep(0.1)

#             # 停止移动
#             self.interface.set_vel(lin_vel=0.0, ang_vel=0.0)

# if __name__ == "__main__":

#     # 这里的interface会自动的初始化和设置所有的nodes，因此不需要rospy.init_node
#     interface = miro.lib.RobotInterface()

#     # 创建 SimpleMovementAndLight 对象并执行
#     controller = happy_mood_motion(interface)
#     controller.spin()
#################################################################################

import rospy
import miro2 as miro
import numpy as np
import time
import math
import threading
from temp_audio_play import AudioPlayback
from Listening_to_generate_respond import generate_single_respond

class HappyMoodFeedback:
    def __init__(self, interface):
        self.interface = interface
        self.pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.total_angle = 0.0  # 累积旋转角度（弧度）
        self.last_angle = 0.0  # 上一次记录的角度（弧度）
        self.subscriber = None

    def normalize_angle(self, angle):
        """
        将角度归一化到 [0, 2π)
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def callback_sensors_update_pose_angle(self, msg):
        """
        回调函数：处理传感器数据并更新机器人位置。
        这里的作用实际上是在手动更新机器人的pose
        由于传感器数据（线速度和角速度）只有这两个，因此需要手动计算更新
        """
        # 获取当前的角速度
        angular_velocity = msg.odom.twist.twist.angular.z

        # 时间间隔
        T = 0.02  # 假设传感器每隔 20ms 发布一次数据

        # 更新机器人朝向（theta）
        current_angle = self.pose[2] + angular_velocity * T
        self.pose[2] = self.normalize_angle(current_angle)

        # 计算当前步长的角度变化（考虑角度环绕问题）
        delta_angle = self.pose[2] - self.last_angle
        if delta_angle < -np.pi:  # 从正值跳负值
            delta_angle += 2 * np.pi
        elif delta_angle > np.pi:  # 从负值跳正值
            delta_angle -= 2 * np.pi

        # 累加角度变化
        self.total_angle += abs(delta_angle)
        self.last_angle = self.pose[2]

        # 输出当前累积角度（以度为单位）
        print(f"Total angle: {np.degrees(self.total_angle):.2f} degrees")
        # print(self.pose)

    def spin(self):
        """
        控制机器人旋转一圈（基于传感器数据）。
        """
        # 设置线速度和角速度
        lin_vel = 0.02
        ang_vel = 0.785
        self.interface.set_vel(lin_vel=lin_vel, ang_vel=ang_vel)

        # 订阅传感器数据
        topic_base_name = "/" + rospy.get_param("MIRO_ROBOT_NAME", "miro")
        topic = topic_base_name + "/sensors/package"
        print(f"Subscribing to {topic}")
        # 自动回调的地方，即一旦有信息发布到"/sensors/package"这个topic，这里就会执行上面定义的更新pose[2]的回调函数，为什么是这个topic？client_map里就是这么写的
        self.subscriber = rospy.Subscriber(topic, miro.msg.sensors_package, self.callback_sensors_update_pose_angle)

        # 等待旋转完成
        rate = rospy.Rate(50)  # 每秒 50 次，被用在下面的while循环中的sleep上了
        target_angle = 2 * np.pi  # 目标角度：一圈（弧度）
        while not rospy.is_shutdown():
            # 检查是否旋转了一圈
            if self.total_angle >= target_angle:
                break
            rate.sleep()
            # 为什么停当前线程一段时间（这里是 1/50 = 0.02 秒）。而且这个数字是根据T = 0.02来的，这个T是什么？是传感器每隔20ms发布一次数据。
            # 20ms也是client_map给的
            # 在暂停期间，ROS 系统可以处理其他任务，比如回调函数 callback_sensors。
            # 确保主循环不会因为快速运行而抢占 ROS 系统的资源。

        # 停止机器人
        self.total_angle = 0.0  # 重置累积角度
        self.stop_spin()

    def stop_spin(self):
        """
        停止机器人所有运动。
        """
        # 停止移动
        self.interface.set_vel(lin_vel=0.0, ang_vel=0.0)
        print("Rotation complete. Robot stopped.")

        # 取消订阅
        if self.subscriber:
            self.subscriber.unregister()
            self.subscriber = None

    ##################################################################################################################
    # 下面是灯光控制的代码
    def light_function(self):
        colors = [
            (255, 0, 0),   # 红色
            (255, 64, 0),  # 橙红色
            (255, 128, 0), # 橙色
            (255, 192, 0), # 橙黄色
            (255, 255, 0)  # 黄色
        ]

        # 让灯光以前中后组交替闪烁 5 秒钟
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
    ##################################################################################################################

    # 下面是尾巴的控制代码
    def tail_function(self, duration=8, frequency=4, amplitude=1.0, default_wag=0.5, default_droop=0.5):
        """
        控制机器人尾巴的摆动功能，模拟狗狗在快乐情绪下的尾巴动作。

        参数:
        - duration (float): 摆动持续时间，单位为秒。
        表示尾巴持续摆动的时间。例如，如果设置为 8，尾巴将摆动 8 秒钟。

        - frequency (float): 摆动频率，单位为每秒摆动次数。
        表示尾巴左右摆动的速度。例如，frequency=1 表示尾巴每秒完成一个来回摆动。

        - amplitude (float): 摆动幅度，范围在 [0.0, 1.0] 之间。
        表示尾巴摆动的幅度大小。例如，amplitude=1.0 表示尾巴从最左摆动到最右（全幅度），
        amplitude=0.5 表示尾巴摆动范围缩小为全幅度的一半。

        - default_wag (float): 尾巴摆动停止后的默认左右位置，范围在 [0.0, 1.0] 之间。
        表示尾巴在停止后回到的位置，例如 default_wag=0.5 表示尾巴居中。

        - default_droop (float): 尾巴摆动停止后的默认上下位置，范围在 [0.0, 1.0] 之间。
        表示尾巴在停止后回到的高度位置，例如 default_droop=0.5 表示尾巴在中等高度。
        0.0 表示尾巴完全抬起，1.0 表示尾巴完全放下。

        功能:
        - 根据正弦函数控制尾巴左右平滑摆动，实现自然的尾巴动作。
        - 动作完成后，将尾巴恢复到指定的默认位置。

        额外说明:
        这里使用的set_cos函数是api用来控制COS关节的，COS关节包含那些可以去interface里面找
        """
        import time
        start_time = time.time()
        while time.time() - start_time < duration:
            wag_angle = 0.5 + amplitude * 0.5 * math.sin(2 * math.pi * frequency * (time.time() % 1))
            self.interface.set_cos(miro.constants.JOINT_WAG, wag_angle)
            self.interface.set_cos(miro.constants.JOINT_DROOP, 0.5)
            time.sleep(0.02)

        # 恢复默认状态
        self.interface.set_cos(miro.constants.JOINT_WAG, default_wag)
        self.interface.set_cos(miro.constants.JOINT_DROOP, default_droop)
   ##################################################################################################################

    def ear_flap_function(self, duration=3, frequency=4, amplitude=0.5):
        """
        控制耳朵上下摆动，模拟情绪反应。
        - duration: 持续时间（秒）。
        - frequency: 摆动频率（每秒）。
        - amplitude: 摆动幅度（0.0 到 1.0）。
        """
        import math, time
        start_time = time.time()
        while time.time() - start_time < duration:
            # 上下摆动
            ear_angle = 0.5 + amplitude * math.sin(2 * math.pi * frequency * (time.time() % 1))
            self.interface.set_cos(miro.constants.JOINT_EAR_L, ear_angle)
            self.interface.set_cos(miro.constants.JOINT_EAR_R, ear_angle)
            time.sleep(0.02)  # 每 20ms 更新

        
    def eye_blink_function(self, duration=5, blink_interval=1.0):
        """
        模拟机器人眨眼。
        - duration: 持续时间（秒）。
        - blink_interval: 每次眨眼的间隔时间（秒）。
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            # 闭眼
            self.interface.set_cos(miro.constants.JOINT_EYE_L, 1.0)
            self.interface.set_cos(miro.constants.JOINT_EYE_R, 1.0)
            time.sleep(0.1)  # 闭眼 0.1 秒

            # 睁眼
            self.interface.set_cos(miro.constants.JOINT_EYE_L, 0.0)
            self.interface.set_cos(miro.constants.JOINT_EYE_R, 0.0)
            time.sleep(blink_interval)  # 等待下一次眨眼
    #############################################################################################################################   
    # 下面是关于头部关节的控制
    # 这里就是KIN_JOINTS 的控制代码了
    # 先看看可以设定的范围
    def print_kin_joint_limits(self):
        print("Kinematic Joint Limits:")
        print(f"JOINT_LIFT: [{miro.constants.LIFT_RAD_MIN}, {miro.constants.LIFT_RAD_MAX}] radians")
        # JOINT_LIFT:0.14-1, 代表头的抬起和低头的范围
        print(f"JOINT_PITCH: [{miro.constants.PITCH_RAD_MIN}, {miro.constants.PITCH_RAD_MAX}] radians")
        # self.interface.set_kin(miro.constants.JOINT_YAW, -0.5)
        # JOINT_YAW: 左右转头的范围（-0.95到0.95）
        print(f"JOINT_YAW: [{miro.constants.YAW_RAD_MIN}, {miro.constants.YAW_RAD_MAX}] radians")
        # self.interface.set_kin(miro.constants.JOINT_PITCH, -0.38)
        # JOINT_PITCH: 仰头和低头的范围（-0.38（朝上）到0.13（朝下看））

    # def head_nod_function(self, duration=5, frequency=1):
    #     """
    #     控制机器人的头部上下点头，同时结合仰头动作，表达快乐情绪。

    #     :param duration: 点头动作持续时间（秒）
    #     :param frequency: 点头频率（次/秒）
    #     """
    #     # JOINT_LIFT 范围：0.14（低头）到 1.0（抬头）
    #     lift_min = 0.14
    #     lift_max = 1.0

    #     # PITCH 范围：-0.38（仰头）到 0.13（低头）
    #     pitch_angle = -0.38  # 直接固定仰头位置

    #     # 设置仰头的 PITCH 角度
    #     self.interface.set_kin(miro.constants.JOINT_PITCH, pitch_angle)

    #     # 开始点头时间
    #     start_time = time.time()
    #     while time.time() - start_time < duration:
    #         # 使用正弦函数生成点头 lift 角度
    #         lift_angle = lift_min + (lift_max - lift_min) / 2 * (1 + math.sin(2 * math.pi * frequency * (time.time() % 1)))
            
    #         # 设置头部 LIFT 角度
    #         self.interface.set_kin(miro.constants.JOINT_LIFT, lift_angle)

    #         # 稍作延迟，模拟平滑的点头动作
    #         time.sleep(0.02)

    #     # 恢复默认头部位置
    #     self.interface.set_kin(miro.constants.JOINT_LIFT, 0.5)  # 默认位置中间
    #     self.interface.set_kin(miro.constants.JOINT_PITCH, 0.0)  # 恢复水平位置

    # 有点吓人于是我决定只用仰头的动作
    def head_high(self, duration=5):
        self.interface.set_kin(miro.constants.JOINT_PITCH, -0.38)
        time.sleep(duration)
        self.interface.set_kin(miro.constants.JOINT_PITCH, 0.0)

    #############################################################################################################################
    # 下面是语音回复的代码
    def happy_audio(self, sound_file):
        playback = AudioPlayback(self.interface, "/miro", sound_file)
        rospy.sleep(2)
        playback.play()

    




def execute_all_functions(controller, duration=5):
    """
    多线程同时执行所有功能。
    - controller: HappyMoodFeedback 类的实例。
    - duration: 动作持续时间（秒）。
    """

    # 创建线程
    spin_thread = threading.Thread(target=controller.spin)
    light_thread = threading.Thread(target=controller.light_function)
    tail_thread = threading.Thread(target=controller.tail_function, args=(duration, 3, 0.8))
    ear_thread = threading.Thread(target=controller.ear_flap_function, args=(duration, 2, 0.5))
    eye_thread = threading.Thread(target=controller.eye_blink_function, args=(duration, 1.0))
    head_thread = threading.Thread(target=controller.head_high, args=(duration,))
    sound_thread = threading.Thread(target=controller.happy_audio, args=("/home/yufeng/mdk/output1.wav",))

    # 启动线程
    sound_thread.start()
    spin_thread.start()
    light_thread.start()
    tail_thread.start()
    ear_thread.start()
    eye_thread.start()
    head_thread.start()


    # 等待线程完成
    sound_thread.join()
    spin_thread.join()
    light_thread.join()
    tail_thread.join()
    ear_thread.join()
    eye_thread.join()
    head_thread.join()

if __name__ == "__main__":
    # 假设已经初始化了 interface 对象
    interface = miro.lib.RobotInterface()

    # 创建控制器对象
    controller = HappyMoodFeedback(interface)

    # # 执行所有功能
    # execute_all_functions(controller, duration=9)
    while not rospy.is_shutdown():
        sound_file = generate_single_respond()
        if sound_file is not None:
            # controller.happy_audio(sound_file)
            execute_all_functions(controller, duration=9)





# if __name__ == "__main__":

#     # 假设已经初始化了 interface 对象
#     interface = miro.lib.RobotInterface()

#     # 创建对象并执行
#     controller = HappyMoodFeedback(interface)
#     # controller.spin()
#     # controller.light_function()
#     # controller.tail_function()
#     # controller.ear_flap_function()
#     # controller.eye_blink_function()
#     # controller.print_kin_joint_limits()
#     # controller.head_nod_function()
#     controller.head_high()
