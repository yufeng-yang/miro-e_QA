import mediapipe as mp
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image

# Miroe相关的包
import rospy
import miro2 as miro
import numpy as np

# 初始化全局变量
detections = []

# mediapipe的基本配置
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    # result的结果为DetectionResult(
    # detections=
    # [Detection(bounding_box=BoundingBox(origin_x=284, origin_y=198, width=202, height=202), 
    # categories=[Category(index=0, score=0.8555264472961426, display_name=None, category_name=None)], 
    # keypoints=[NormalizedKeypoint(x=0.49986565113067627, y=0.5514069199562073, label='', score=0.0), 
    # NormalizedKeypoint(x=0.6363276839256287, y=0.5330133438110352, label='', score=0.0), 
    # NormalizedKeypoint(x=0.5552176237106323, y=0.6522853374481201, label='', score=0.0), 
    # NormalizedKeypoint(x=0.5729540586471558, y=0.7331933379173279, label='', score=0.0), 
    # NormalizedKeypoint(x=0.4650926887989044, y=0.581007182598114, label='', score=0.0), 
    # NormalizedKeypoint(x=0.7519003748893738, y=0.5406632423400879, label='', score=0.0)])])

    # 从结果而言似乎是一个叫做detections的list，里面包含了检测到的人脸的信息
    # 但是好像这里作为回掉函数，无法直接return出去。

    global detections
    detections = result.detections  # 保存检测结果到全局变量
    # 这里使用=直接去赋值，而不是append，从而做到只保存一帧率的检测结果


options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='/home/yufeng/mdk/bin/trained_model/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    # 由于我这里使用了LIVE_STREAM模式，因此默认需要一个回调函数。
    result_callback=print_result)






class CameraDisplay:
    def __init__(self):
        self.interface = miro.lib.RobotInterface()

    def align_face_to_center(self, bbox, frame):
        # 获取人脸中心点
        face_center_x = bbox.origin_x + bbox.width / 2
        face_center_y = bbox.origin_y + bbox.height / 2
        # 在画面上绘制人脸中心点（绿色）和画面中心点（红色）


        # 获取画面中心点
        frame_center_x = frame.shape[1] / 2
        frame_center_y = frame.shape[0] / 2

        # 计算偏差
        dx = face_center_x - frame_center_x
        dy = face_center_y - frame_center_y

        # 转换为角度调整
        gain_x, gain_y = 0.005, 0.005
        yaw_adjustment = -dx * gain_x
        pitch_adjustment = dy * gain_y

        # 获取当前头部角度
        kin_joints = self.interface.msg_kin_joints.get()
        current_yaw = kin_joints.position[miro.constants.JOINT_YAW]
        current_pitch = kin_joints.position[miro.constants.JOINT_PITCH]

        # # 调整头部
        # self.interface.set_kin(miro.constants.JOINT_YAW, current_yaw + yaw_adjustment)
        # self.interface.set_kin(miro.constants.JOINT_PITCH, current_pitch + pitch_adjustment)
        
    def loop(self):
        with FaceDetector.create_from_options(options) as detector:
            while not rospy.core.is_shutdown():
                left_image_frame = self.interface.get_cameras()[0]
                if left_image_frame is not None:
                    frame = left_image_frame.data
                    if isinstance(frame, np.ndarray):
                        # 将图像转换为 MediaPipe 格式
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.copy())
                        frame_timestamp_ms = int(rospy.Time.now().to_nsec() / 1e6)
                        detector.detect_async(mp_image, frame_timestamp_ms)

                        if len(detections) > 0:
                            # 找到置信度最高的人脸
                            best_detection = max(detections, key=lambda d: d.categories[0].score)
                            bbox = best_detection.bounding_box

                            # 计算人脸中心点
                            face_center_x = int(bbox.origin_x + bbox.width / 2)
                            face_center_y = int(bbox.origin_y + bbox.height / 2)

                            # 画面中心点
                            frame_center_x = int(frame.shape[1] / 2)
                            frame_center_y = int(frame.shape[0] / 2)

                            # 定义绿色框的范围
                            box_width = int(frame.shape[1] * 0.5)  # 宽度为画面宽度的30%
                            box_height = int(frame.shape[0] * 0.5)  # 高度为画面高度的30%
                            box_x_min = frame_center_x - box_width // 2
                            box_x_max = frame_center_x + box_width // 2
                            box_y_min = frame_center_y - box_height // 2
                            box_y_max = frame_center_y + box_height // 2

                            # 绘制绿色框
                            cv2.rectangle(frame, (box_x_min, box_y_min), (box_x_max, box_y_max), (0, 255, 0), 2)

                            # 绘制人脸中心点（红色）
                            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)  # 红色点

                            # 检查人脸中心是否在绿色框内
                            if box_x_min <= face_center_x <= box_x_max and box_y_min <= face_center_y <= box_y_max:
                                # 人脸在绿色框内，不进行调整
                                print("Face within green box, no adjustment needed.")
                            else:
                                # 人脸不在绿色框内，进行调整
                                dx = face_center_x - frame_center_x
                                dy = face_center_y - frame_center_y
                                gain_x, gain_y = 0.005, 0.005
                                yaw_adjustment = -dx * gain_x
                                pitch_adjustment = dy * gain_y

                                # 获取当前头部角度
                                kin_joints = self.interface.msg_kin_joints.get()
                                current_yaw = kin_joints.position[miro.constants.JOINT_YAW]
                                current_pitch = kin_joints.position[miro.constants.JOINT_PITCH]

                                # 计算新目标角度
                                new_yaw = current_yaw + yaw_adjustment
                                new_pitch = current_pitch + pitch_adjustment

                                # 设置新的角度
                                self.interface.set_kin(miro.constants.JOINT_YAW, new_yaw)
                                self.interface.set_kin(miro.constants.JOINT_PITCH, new_pitch)

                        # 显示摄像头画面
                        cv2.imshow('Miro Left Camera', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break



if __name__ == "__main__":
    display = CameraDisplay()
    display.loop()