# import rospy
# from sensor_msgs.msg import CompressedImage
# import cv2
# from cv_bridge import CvBridge, CvBridgeError
# import numpy as np
# from ultralytics import YOLO
# import os
# import miro2 as miro

# class ObjectDetectionClient:
#     def __init__(self, target_label="person", confidence_threshold=0.5, detection_interval=1):
#         # 初始化 YOLO 模型
#         self.model = YOLO("yolo11s.pt")
#         self.target_label = target_label
#         self.confidence_threshold = confidence_threshold
#         self.detection_interval = detection_interval
#         self.frame_count = 0
#         self.event_triggered = False  # 用于标记事件是否已触发
        
#         # ROS -> OpenCV 图像转换器
#         self.bridge = CvBridge()

#         # 初始化相机数据存储
#         self.left_camera_image = None

#         # 机器人接口，用于控制运动或执行其他动作
#         self.interface = miro.lib.RobotInterface()

#         # 订阅左眼摄像头的压缩图像
#         topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
#         self.sub_caml = rospy.Subscriber(topic_base_name + "/sensors/caml/compressed",
#                                          CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)

#     def callback_caml(self, ros_image):
#         try:
#             # 转换为 OpenCV 格式的图像
#             image = self.bridge.compressed_imgmsg_to_cv2(ros_image, "bgr8")
#             self.left_camera_image = image

#             # 每隔 detection_interval 帧检测一次
#             if self.frame_count % self.detection_interval == 0 and not self.event_triggered:
#                 results = self.model(image)
#                 for result in results:
#                     boxes = result.boxes.data
#                     for box in boxes:
#                         label = result.names[int(box[-1].item())]
#                         confidence = box[4].item()

#                         # 检测到目标标签且置信度高于阈值
#                         if label == self.target_label and confidence > self.confidence_threshold:
#                             x1, y1, x2, y2 = map(int, box[:4])
#                             # 在图像中标记检测到的目标
#                             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                             cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                             # 触发事件
#                             self.trigger_event()
#                             self.event_triggered = True  # 设置事件标志为已触发

#             # 显示图像窗口
#             cv2.imshow("Left Camera - Object Detection", image)
#             cv2.waitKey(1)

#             # 增加帧计数器
#             self.frame_count += 1

#         except CvBridgeError as e:
#             print(f"CvBridge Error: {e}")

#     def trigger_event(self):
#         """
#         定义检测到目标后触发的事件。
#         """
#         print("目标人物已检测到，触发事件！")
#         # 例如，可以让机器人停止运动或播放声音
#         self.interface.set_vel(0.0, 0.0)
#         # 其他事件逻辑，例如发送 ROS 消息或控制其他模块

#     def loop(self):
#         # 保持 ROS 节点运行
#         rospy.spin()


# if __name__ == "__main__":
#     # 初始化 ROS 节点
#     rospy.init_node("object_detection_client", anonymous=True)

#     # 创建检测对象，并启动循环
#     client = ObjectDetectionClient(target_label="person", confidence_threshold=0.5, detection_interval=1)
#     client.loop()



import cv2
from ultralytics import YOLO
import threading

class ObjectDetectionClient:
    def __init__(self, target_label="person", confidence_threshold=0.5, detection_interval=10):
        # 初始化 YOLO 模型
        self.model = YOLO("yolo11s.pt")  # 请确保模型权重路径正确
        self.target_label = target_label
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.event_triggered = False  # 用于标记事件是否已触发
        
        # 打开电脑摄像头
        self.cap = cv2.VideoCapture(0)  # 0 是默认摄像头编号

        if not self.cap.isOpened():
            print("无法打开摄像头")
            exit()

    def process_frame(self):
        """
        处理视频帧并检测目标。
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break

            # 每隔 detection_interval 帧检测一次
            if self.frame_count % self.detection_interval == 0 and not self.event_triggered:
                results = self.model(frame)
                for result in results:
                    boxes = result.boxes.data
                    for box in boxes:
                        label = result.names[int(box[-1].item())]
                        confidence = box[4].item()

                        # 检测到目标标签且置信度高于阈值
                        if label == self.target_label and confidence > self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box[:4])
                            # 在图像中标记检测到的目标
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # 触发事件
                            self.trigger_event()

            # 显示图像窗口
            cv2.imshow("Webcam - Object Detection", frame)

            # 按键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 增加帧计数器
            self.frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def trigger_event(self):
        """
        定义检测到目标后触发的事件。
        """
        print("目标人物已检测到，触发事件！")
        self.event_triggered = True  # 设置事件标志为已触发
        # 可以使用延时或其他逻辑来恢复触发状态
        self.reset_event_flag(delay=10)

    def reset_event_flag(self, delay=10):
        """
        延迟重置事件标志。
        """
        def reset():
            self.event_triggered = False
            print("事件标志已重置，准备再次触发事件。")
        threading.Timer(delay, reset).start()


if __name__ == "__main__":
    # 创建检测对象，并启动检测
    client = ObjectDetectionClient(target_label="person", confidence_threshold=0.5, detection_interval=10)
    client.process_frame()

