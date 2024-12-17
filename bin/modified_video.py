import rospy
from sensor_msgs.msg import CompressedImage
import time
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError

class VideoTarget:
    def __init__(self, mode="record", index_to_record=1):
        """
        初始化 ClientVideo 类
        :param mode: 模式，可以是 "show" 或 "record"
        :param index_to_record: 指定要处理的摄像头索引 (0: 左摄像头, 1: 右摄像头)
        """
        self.mode = mode
        self.index_to_record = index_to_record
        self.input_camera = [None, None]

        # ROS -> OpenCV 转换器
        self.image_converter = CvBridge()

        # 订阅 ROS 摄像头话题
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME", "default_robot")
        self.sub_caml = rospy.Subscriber(topic_base_name + "/sensors/caml/compressed",
                                         CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
        self.sub_camr = rospy.Subscriber(topic_base_name + "/sensors/camr/compressed",
                                         CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)

        # 报告
        print(f"Initialized in {self.mode} mode for camera index {index_to_record}")

    def callback_cam(self, ros_image, index):
        """
        回调函数，用于处理摄像头图像
        """
        try:
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "bgr8")
            self.input_camera[index] = image
        except CvBridgeError:
            pass

    def callback_caml(self, ros_image):
        """
        左摄像头回调
        """
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):
        """
        右摄像头回调
        """
        self.callback_cam(ros_image, 1)

    def loop(self):
        """
        主循环，用于处理图像并根据模式执行操作
        """
        outfile = None
        outcount = 0
        t0 = time.time()

        while not rospy.core.is_shutdown():
            index = self.index_to_record  # 指定的摄像头索引
            image = self.input_camera[index]

            if image is not None:
                self.input_camera[index] = None

                # 显示模式
                if self.mode == "show":
                    cv2.imshow(f"client_video: camera {index}", image)
                    cv2.waitKey(1)

                if self.mode == "record":
                    # 如果输出文件未初始化
                    if outfile is None:
                        # 假设初始帧间隔
                        prev_time = time.time()
                        fps = 15  # 初始默认帧率

                        outfile = cv2.VideoWriter(
                            f'client_video_camera_{index}.avi',
                            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            fps, (image.shape[1], image.shape[0])
                        )
                    else:
                        # 动态计算 FPS
                        current_time = time.time()
                        time_interval = current_time - prev_time
                        prev_time = current_time

                        if time_interval > 0:  # 避免除以零
                            fps = 1 / time_interval
                            # print(f"Dynamic FPS: {fps:.2f}")

                    # 写入当前帧到视频文件
                    outfile.write(image)
                    outcount += 1


                    # 报告
                    t1 = time.time()
                    if (t1 - t0) > 1.0:
                        t0 += 1.0
                        print(f"Frames recorded so far: {outcount}")

            time.sleep(0.02)

        # 释放资源
        if outfile is not None:
            outfile.release()

if __name__ == "__main__":
    rospy.init_node("client_video", anonymous=True)
    client = VideoTarget(mode="record", index_to_record=1)
    client.loop()
