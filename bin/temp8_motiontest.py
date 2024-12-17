import rospy
import threading
import time
import miro2 as miro

class SimpleMovementAndLight:
    def __init__(self, interface):
        self.interface = interface

    def spin(self, duration=8):

        # 设置线速度和角速度
        lin_vel = 0.2  # 设置较低线速度，表示小幅前进
        ang_vel = 0.785 # 设置角速度，表示旋转

        for _ in range(int(duration / 0.1)):
            self.interface.set_vel(lin_vel=lin_vel, ang_vel=ang_vel)
            time.sleep(0.1)

        # 停止移动
        self.interface.set_vel(lin_vel=0.0, ang_vel=0.0)

    def light_effect(self):
        """控制灯光闪烁"""
        colors = [
            (255, 0, 0),   # 红色
            (255, 255, 0), # 黄色
            (0, 255, 0),   # 绿色
        ]

        # 灯光交替闪烁
        for i in range(10):
            color = colors[i % len(colors)]
            brightness = 200 if i % 2 == 0 else 50  # 每次闪烁时亮度变化
            self.interface.set_illum(miro.constants.ILLUM_FRONT, color, brightness)
            time.sleep(0.5)

        # 设置稳定的黄色
        self.interface.set_illum(miro.constants.ILLUM_ALL, (255, 255, 0), 100)

    def main(self):
        # 创建线程处理移动和灯光控制
        move_thread = threading.Thread(target=self.spin)
        light_thread = threading.Thread(target=self.light_effect)

        # 启动线程
        move_thread.start()
        light_thread.start()

        # 等待线程完成
        move_thread.join()
        light_thread.join()

if __name__ == "__main__":
    # rospy.init_node("simple_movement_and_light")

    # 假设已经初始化了 interface 对象
    interface = miro.lib.RobotInterface()

    # 创建 SimpleMovementAndLight 对象并执行
    controller = SimpleMovementAndLight(interface)
    controller.main()
