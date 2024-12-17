import mediapipe as mp
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image

# Miroe相关的包
import rospy
import miro2 as miro
import numpy as np

# 这是没有被我魔改过的原版DDAMFN_NETWORK.py
# from DDAMFN_NETWORK import DDAMNet

# 这是我魔改过的DDAMFN_NETWORK.py
from DDAM import DDAMNet
from extract_video_feature_tools import extrace_dealt_feature, feature_to_7probability

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


# 下面是表情识别的部分准备
# 表情类别
class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']
# 导入表情识别模型的方法，将会return模型和CPU
def load_emotion_model(model_path, num_class=7, num_head=2):
    # 好像因为版本问题，目前cuda不可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DDAMNet(num_class=num_class, num_head=num_head, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model,device

# 该表情识别模型，对于输入的图片需要额外的预处理
def preprocess_image(image):
    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    img = data_transforms_val(img)
    img = img.unsqueeze(0)  # 增加一个维度，模拟batch
    return img

# 进行表情识别(仅输出概率)的版本
# def predict_emotion(face_image, model, device):
#     # img = preprocess_image(face_image).to(device)
#     with torch.no_grad():
#         out, _, _ = model(face_image)
#         probabilities = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
#     return probabilities

# 进行表情识别(输出概率和特征)的版本
def predict_emotion(face_image, model, device, return_feature=False):
    # 确保输入已预处理并在正确的设备上
    face_image = face_image.to(device)
    
    with torch.no_grad():
        # 调用模型的 forward 方法，取决于是否需要返回特征
        if return_feature:
            out, _, _, feature = model(face_image, return_dealt_feature=True)
        else:
            out, _, _ = model(face_image)
            feature = None  # 不需要特征时，默认设为 None
        
        # 计算分类概率
        probabilities = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    
    # 返回概率和特征
    return probabilities, feature


emotion_model,device = load_emotion_model('/home/yufeng/mdk/bin/affecnet7_epoch19_acc0.671.pth')

# 打标签
def format_emotion_result(probabilities):
    selected_indices = [class_names.index('Neutral'), 
                        class_names.index('Happy'), 
                        class_names.index('Sad'), 
                        class_names.index('Angry'),
                        class_names.index('Surprise'),
                        class_names.index('Fear'),]
    selected_probabilities = {class_names[i]: probabilities[i] for i in selected_indices}
    # 按概率值从高到低排序，并只保留前两项
    sorted_probabilities = dict(sorted(selected_probabilities.items(), key=lambda item: item[1], reverse=True)[:2])
    return sorted_probabilities

# 可视化结果
def draw_result(frame, sorted_probabilities, x, y, w, h):
    # 在人脸上绘制表情标签
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绿色边框

    # 不显示概率值
    # emotion_text = ", ".join([f"{key}: {value:.2f}" for key, value in sorted_probabilities.items()])
    # cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame



# with FaceDetector.create_from_options(options) as detector:
#     cap = cv2.VideoCapture(0)

#     # 在主循环中处理检测和绘制逻辑
#     while True:
#         ret, frame = cap.read()  # 读取摄像头帧

#         if not ret:
#             break

#         # 转换图像为 MediaPipe 格式
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.copy())
#         frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
#         # 执行异步检测
#         detector.detect_async(mp_image, frame_timestamp_ms)
#         # 获取检测结果
#         # 这里好像收到异步的影响，无法直接获取检测结果，只能使用for loop，因为for loop在detections没有被更新前，甚至不会启动遍历。
#         for detection in detections:
#             bbox = detection.bounding_box
#             # 有了人脸的边框就可以保存这些数值
#             x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
#             # 提取人脸区域，将这部分作为表情识别的input
#             # 确保裁剪区域在图像范围内
#             x, y = max(0, x), max(0, y)
#             w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)

#             face_image = frame[y:y+h, x:x+w]

#             # ==========================================================
#             # 测试代码
#             # 显示frame
#             # cv2.imshow('face', face_image)
#             # 实时刷新
#             # cv2.waitKey(1)
#             # ==========================================================

#             # 进行表情识别,首先是预处理
#             if face_image.size > 0:
#                 face_image = preprocess_image(face_image)
#                 # 进行表情识别
#                 probabilities = predict_emotion(face_image, emotion_model, device)
#                 # 打标签
#                 sorted_probabilities = format_emotion_result(probabilities)
#                 # 可视化结果
#                 # 这里直接用frame赋值，为了在没有人脸的时候，也正常显示摄像头画面
#                 frame = draw_result(frame, sorted_probabilities, x, y, w, h)
#             cv2.imshow('face', frame)
#         # 刷新代码和按q退出
#         if cv2.waitKey(1) == ord('q'):
#             break


# class CameraDisplay:
#     def __init__(self):
#         self.interface = miro.lib.RobotInterface()
        
#         # 用于存储feature
#         self.accunlated_feature = None
        
#     def result_loop(self):
#            with FaceDetector.create_from_options(options) as detector:
#             while not rospy.core.is_shutdown():
#                 left_image_frame = self.interface.get_cameras()[0]
#                 if left_image_frame is not None:
#                     frame = left_image_frame.data
#                     if isinstance(frame, np.ndarray):
#                         # 将图像转换为 MediaPipe 格式
#                         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.copy())
#                         frame_timestamp_ms = int(rospy.Time.now().to_nsec() / 1e6)
#                         detector.detect_async(mp_image, frame_timestamp_ms)

#                         if len(detections) > 0:
#                             # 找到置信度最高的检测结果
#                             best_detection = max(detections, key=lambda d: d.categories[0].score)

#                             bbox = best_detection.bounding_box
#                             # 原始边框
#                             x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

#                             # 延伸边框周边 20 像素
#                             padding = 10
#                             x = max(0, x - padding)  # 左边界减去 padding，确保不超出图像范围
#                             y = max(0, y - padding)  # 上边界减去 padding
#                             w = min(frame.shape[1] - x, w + 2 * padding)  # 宽度增加 padding，但不超出图像宽度
#                             h = min(frame.shape[0] - y, h + 2 * padding)  # 高度增加 padding，但不超出图像高度)

#                             face_image = frame[y:y + h, x:x + w]

#                             if face_image.size > 0:
#                                 face_image = preprocess_image(face_image)
#                                 # ==========================
#                                 # 仅输出概率的版本
#                                 # probabilities = predict_emotion(face_image, emotion_model, device)
#                                 # 既输出概率又输出特征的版本
#                                 probabilities, frame_feature = predict_emotion(face_image, emotion_model, device, return_feature=True)
#                                 print(frame_feature)

#                                 # ==========================
#                                 sorted_probabilities = format_emotion_result(probabilities)
#                                 frame = draw_result(frame, sorted_probabilities, x, y, w, h)
#                                 # 调用 ui_loop，传递 frame 和表情概率
#                                 # self.ui_loop(sorted_probabilities)

#                                 # self.video_loop_without_probabilities(frame, sorted_probabilities)

#                                 self.ui_video_loop_without_prob_and_extend(frame, sorted_probabilities)
                                

#                         # 显示摄像头画面
#                         cv2.imshow('Miro Left Camera', frame)
#                         if cv2.waitKey(1) & 0xFF == ord('q'):
#                             break
class CameraDisplay:
    def __init__(self, interface, robot_eye_show=False):
        self.interface = interface
        # 多设置了一个参数，用于控制是否显示机器人识别的图像
        self.isshow = robot_eye_show

        # 新增属性
        self.accumulated_feature = None  # 累加的特征
        self.feature_count = 0  # 累加的帧数
        self.accumulation_interval = 5  # 累加间隔，可按需调整

    def result_loop(self):
        """
        实时检测人脸，并每 n 帧累加一次特征
        """
        if self.isshow == False:
            print("Display is disabled. Skipping result loop.")
            return


        if self.isshow == True:
            with FaceDetector.create_from_options(options) as detector:
                while not rospy.core.is_shutdown():
                # 这里不采用rospy 是否run作为控制，而是采用特定的控制变量
                    left_image_frame = self.interface.get_cameras()[0]
                    if left_image_frame is not None:
                        frame = left_image_frame.data
                        if isinstance(frame, np.ndarray):
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.copy())
                            frame_timestamp_ms = int(rospy.Time.now().to_nsec() / 1e6)
                            detector.detect_async(mp_image, frame_timestamp_ms)

                            if len(detections) > 0:
                                best_detection = max(detections, key=lambda d: d.categories[0].score)
                                bbox = best_detection.bounding_box
                                x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                                padding = 10
                                x, y = max(0, x - padding), max(0, y - padding)
                                w, h = min(frame.shape[1] - x, w + 2 * padding), min(frame.shape[0] - y, h + 2 * padding)

                                face_image = frame[y:y + h, x:x + w]

                                if face_image.size > 0:
                                    face_image = preprocess_image(face_image)
                                    probabilities, frame_feature = predict_emotion(face_image, emotion_model, device, return_feature=True)
                                    sorted_probabilities = format_emotion_result(probabilities)
                                    frame = draw_result(frame, sorted_probabilities, x, y, w, h)
                                    # 调用 ui_loop，传递 frame 和表情概率
                                    # self.ui_loop(sorted_probabilities)

                                    # self.video_loop_without_probabilities(frame, sorted_probabilities)

                                    self.ui_video_loop_without_prob_and_extend(frame, sorted_probabilities)

                                    # # 每 n 帧累加一次特征
                                    # 用户输出特征的代码，现在先放弃使用以提高效率
                                    # if self.accumulated_feature is None:
                                    #     self.accumulated_feature = frame_feature
                                    # else:
                                    #     self.accumulated_feature += frame_feature
                                    # self.feature_count += 1

                                    # # 输出累加状态
                                    # if self.feature_count % self.accumulation_interval == 0:
                                    #     print(f"Accumulated Feature at {self.feature_count}")

                            cv2.imshow('Miro Left Camera', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

    def get_accumulated_feature(self):
        """
        返回当前累加的特征并重置状态
        """
        if self.accumulated_feature is not None and self.feature_count > 0:
            average_feature = self.accumulated_feature / self.feature_count
            self.accumulated_feature = None
            self.feature_count = 0
            return average_feature
        return None


    def ui_video_loop(self, frame, probabilities):
        # 定义心情对应颜色
        emotion_colors = {
            "Angry": (0, 0, 255),  # 红色
            "Neutral": (255, 255, 0),  # 黄色
            "Sad": (0, 0, 255),  # 蓝色
            "Happy": (0, 255, 0),  # 绿色
            "Surprise": (0, 255, 0),  # 绿色
        }

        # 获取前两个心情和对应概率
        top_emotions = list(probabilities.items())[:2]
        emotion1, score1 = top_emotions[0]
        emotion2, score2 = top_emotions[1]

        # 计算插值颜色
        color1 = np.array(emotion_colors.get(emotion1, (0, 0, 0)))
        color2 = np.array(emotion_colors.get(emotion2, (0, 0, 0)))
        ratio = score1 / (score1 + score2)
        interpolated_color = (color1 * ratio + color2 * (1 - ratio)).astype(int)

        # 创建背景
        bg_color = np.zeros_like(frame, dtype=np.uint8)
        bg_color[:] = interpolated_color

        # 缩小摄像头画面并叠加
        small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        x_offset = (bg_color.shape[1] - small_frame.shape[1]) // 2
        y_offset = (bg_color.shape[0] - small_frame.shape[0]) // 2
        bg_color[y_offset:y_offset + small_frame.shape[0], x_offset:x_offset + small_frame.shape[1]] = small_frame


        # 显示画面
        cv2.imshow('Emotion Display', bg_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        
    
    def video_loop_without_probabilities(self, frame, probabilities):
        # 定义心情对应颜色
        emotion_colors = {
            "Angry": (0, 0, 255),  # 红色
            "Neutral": (0, 255, 255),  # 黄色
            "Sad": (0, 0, 255),  # 蓝色
            "Happy": (0, 255, 0),  # 绿色
            "Surprise": (0, 255, 0),  # 绿色
            # 其他心情可以继续补充
        }

        # 获取最高概率的心情和对应颜色
        top_emotion, top_score = max(probabilities.items(), key=lambda x: x[1])
        top_color = np.array(emotion_colors.get(top_emotion, (0, 0, 0)))

        # 创建背景
        bg_color = np.zeros_like(frame, dtype=np.uint8)
        bg_color[:] = top_color

        # 缩小摄像头画面到原来的3/4大小并叠加
        new_width = int(frame.shape[1] * 3 / 4)
        new_height = int(frame.shape[0] * 3 / 4)
        small_frame = cv2.resize(frame, (new_width, new_height))
        x_offset = (bg_color.shape[1] - small_frame.shape[1]) // 2
        y_offset = (bg_color.shape[0] - small_frame.shape[0]) // 2
        bg_color[y_offset:y_offset + small_frame.shape[0], x_offset:x_offset + small_frame.shape[1]] = small_frame

        # 显示画面
        cv2.imshow('Emotion Display', bg_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        

    def ui_video_loop_without_prob_and_extend(self, frame, probabilities):
        # 定义心情对应颜色
        emotion_colors = {
            "Angry": (0, 0, 255),  # 红色
            "Neutral": (255, 255, 0),  # 黄色
            "Sad": (255, 0, 0),  # 蓝色
            "Happy": (0, 255, 0),  # 绿色
            "Surprise": (0, 255, 0),  # 绿色
            # 其他心情可以继续补充
        }

        # 获取最高概率的心情和对应颜色
        top_emotion, top_score = max(probabilities.items(), key=lambda x: x[1])
        top_color = np.array(emotion_colors.get(top_emotion, (0, 0, 0)))

        # 拓宽背景，增加边框
        border_size = 50  # 定义边框宽度
        bg_color = cv2.copyMakeBorder(
            frame,
            top=border_size//2,
            bottom=border_size,
            left=border_size//2,
            right=border_size//2,
            borderType=cv2.BORDER_CONSTANT,
            value=top_color.tolist()
        )

            # 添加文字 "Your emotion is (最高的那个结果)"
        text = f"Your emotion is: {top_emotion}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)  # 白色文字

        # 计算文字位置（显示在底部中心）
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (bg_color.shape[1] - text_size[0]) // 2
        text_y = bg_color.shape[0] - 10  # 距离底部 10 像素

        cv2.putText(bg_color, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        # 显示画面
        cv2.imshow('Emotion Display', bg_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

        
    

    def ui_loop(self, probabilities):
        # 定义心情对应颜色
        emotion_colors = {
            "Angry": (0, 0, 255),      # 红色
            "Neutral": (0, 200, 200),  # 黄色
            "Sad": (255,0,0),        # 蓝色
            "Happy": (0, 255, 0),      # 绿色
            "Surprise": (255, 255, 0),   
            # 其他心情可以继续补充
        }

        # 获取前两个心情和对应概率
        top_emotions = list(probabilities.items())[:2]
        emotion1, score1 = top_emotions[0]
        emotion2, score2 = top_emotions[1]

        # 动态计算插值颜色
        color1 = np.array(emotion_colors.get(emotion1, (0, 0, 0)))  # 默认黑色
        color2 = np.array(emotion_colors.get(emotion2, (0, 0, 0)))
        ratio = score1 / (score1 + score2)
        # interpolated_color = (color1 * ratio + color2 * (1 - ratio)).astype(int)
        interpolated_color = (color1).astype(int)
        # interpolated_color = (0,255,0)

        # 创建新的背景
        bg_color = np.zeros((500, 800, 3), dtype=np.uint8)  # 固定窗口大小
        bg_color[:] = interpolated_color

        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 3
        text_color = (255, 255, 255)  # 白色文字

        # 第一行文字：固定 "Your emotion is"
        text1 = "Your emotion is"
        text1_font_scale = 1.5  # 正常大小
        text1_size = cv2.getTextSize(text1, font, text1_font_scale, font_thickness)[0]
        text1_x = (bg_color.shape[1] - text1_size[0]) // 2
        text1_y = (bg_color.shape[0] // 2) - 100  # 向上调整
        cv2.putText(bg_color, text1, (text1_x, text1_y), font, text1_font_scale, text_color, font_thickness)

        # 第二行文字：主情绪（放大字体）
        text2 = emotion1
        text2_font_scale = 3  # 更大字体
        text2_size = cv2.getTextSize(text2, font, text2_font_scale, font_thickness)[0]
        text2_x = (bg_color.shape[1] - text2_size[0]) // 2
        text2_y = (bg_color.shape[0] // 2)  # 居中
        cv2.putText(bg_color, text2, (text2_x, text2_y), font, text2_font_scale, text_color, font_thickness)

        # 第三行文字：次情绪（较小字体）
        text3 = f"({emotion2})"
        text3_font_scale = 1.2  # 较小字体
        text3_size = cv2.getTextSize(text3, font, text3_font_scale, font_thickness)[0]
        text3_x = (bg_color.shape[1] - text3_size[0]) // 2
        text3_y = (bg_color.shape[0] // 2) + 100  # 向下调整
        cv2.putText(bg_color, text3, (text3_x, text3_y), font, text3_font_scale, text_color, font_thickness)


        # 显示画面
        cv2.imshow('Emotion Display', bg_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        



if __name__ == "__main__":
    interface = miro.lib.RobotInterface()
    display = CameraDisplay(interface, True)
    display.result_loop()
    # average_feature = display.get_accumulated_feature()
    # if average_feature is not None:
    #     # print("Final Average Feature:", average_feature)
    #     video_probabilities = feature_to_7probability(emotion_model, average_feature)
    #     print("Video Probabilities:", video_probabilities)






