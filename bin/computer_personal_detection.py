import cv2
from ultralytics import YOLO
import time

from computer_play_sound import play_sound



class ObjectDetectionClient:
    '''用于检测人物发出声音的代码'''
    def __init__(self, target_label="person", confidence_threshold=0.5, detection_interval=5):
        # 初始化 YOLO 模型
        self.model = YOLO("yolo11s.pt")
        self.target_label = target_label
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.event_triggered = False  # 用于标记事件是否已触发

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

    def process_frame(self, frame):
        # 每隔 detection_interval 帧检测一次
        if self.frame_count % self.detection_interval == 0:
            results = self.model(frame)
            for result in results:
                boxes = result.boxes.data
                for box in boxes:
                    label = result.names[int(box[-1].item())]  # 检测到的类别名称
                    confidence = box[4].item()  # 置信度

                    # 解析检测框坐标
                    x1, y1, x2, y2 = map(int, box[:4])

                    # 绘制检测框和类别标签
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 如果检测到目标标签（如 person）并未触发过事件，触发事件
                    if label == self.target_label and confidence > self.confidence_threshold and not self.event_triggered:
                        self.trigger_event()
                        self.event_triggered = True  # 设置事件标志为已触发

        # 显示图像窗口
        cv2.imshow("Webcam - Object Detection", frame)
        cv2.waitKey(1)

        # 增加帧计数器
        self.frame_count += 1

    def trigger_event(self):
        """
        定义检测到目标后触发的事件。
        """
        print("目标人物已检测到，触发事件！")
        # 播放音频
        play_sound('/home/yufeng/mdk/twice_happy_dog_barking_mono_8000.wav')

    def loop(self):
        # 循环读取摄像头帧
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头输入")
                break
            self.process_frame(frame)

        # 释放摄像头资源
        self.cap.release()
        cv2.destroyAllWindows()
