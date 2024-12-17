import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import cv2
# import numpy as np
# import time

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

# # 表情识别模型加载和预测（你可以在这里替换为你的表情识别模型代码）
# def load_emotion_model():
#     # 这里放你的表情识别模型加载代码
#     pass

# def predict_emotion(face_image):
#     # 这里放你的表情识别预测代码
#     # 返回预测的表情类别或置信度
#     pass

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from DDAMFN_NETWORK import DDAMNet

# 表情类别
class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']

import cv2
import numpy as np

def display_emotion_color(probabilities):
    """
    显示一个窗口，背景颜色对应最高概率的情绪颜色。
    """
    # 定义情绪对应的颜色
    emotion_colors = {
        "Neutral": (128, 128, 128),  # 灰色
        "Happy": (0, 255, 0),       # 绿色
        "Sad": (255, 0, 0),         # 蓝色
        "Surprise": (255, 255, 0),  # 黄色
        "Fear": (128, 0, 128),      # 紫色
        "Disgust": (0, 128, 0),     # 深绿色
        "Angry": (255, 0, 0)        # 红色
    }

    # 获取最高概率的情绪
    top_emotion, top_score = max(probabilities.items(), key=lambda x: x[1])
    color = emotion_colors.get(top_emotion, (0, 0, 0))  # 默认黑色

    # 创建一个显示情绪颜色的窗口
    bg_color = np.zeros((300, 300, 3), dtype=np.uint8)
    bg_color[:] = color

    # 添加情绪文字
    text = f"{top_emotion}: {top_score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (bg_color.shape[1] - text_size[0]) // 2
    text_y = (bg_color.shape[0] + text_size[1]) // 2
    cv2.putText(bg_color, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

    # 显示窗口
    cv2.imshow("Emotion Color Display", bg_color)



# 加载表情识别模型
def load_emotion_model(model_path, num_class=7, num_head=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DDAMNet(num_class=num_class, num_head=num_head, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

# 对图像进行预处理
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

# 进行表情识别
def predict_emotion(face_image, model, device):
    img = preprocess_image(face_image).to(device)
    with torch.no_grad():
        out, _, _ = model(img)
        probabilities = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    return probabilities

# 表情预测结果格式化
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


# 回调函数，用于处理检测结果。这是LIVE模式下mediapipe必备的
# def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
#     if result.detections:
#         for detection in result.detections:
#             category = detection.categories[0]
#             confidence = category.score
#             print(f"Detection confidence: {confidence}")

#             # 如果置信度足够高，则提取人脸区域
#             if confidence > 0.65:
#                 bbox = detection.bounding_box
#                 x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

#                 # 转换图像格式为 OpenCV 格式
#                 frame = np.array(output_image.numpy_view())

#                 # 确保裁剪区域在图像范围内
#                 x = max(0, x)
#                 y = max(0, y)
#                 w = min(w, frame.shape[1] - x)
#                 h = min(h, frame.shape[0] - y)

#                 face_image = frame[y:y+h, x:x+w]

#                 # 可视化裁剪的人脸区域
#                 if face_image.size > 0:

#                     # 调用表情识别函数对裁剪的人脸进行表情预测
#                     probabilities = predict_emotion(face_image, model, device)
#                     selected_probabilities = format_emotion_result(probabilities)
#                     print(f"Emotion prediction: {selected_probabilities}")
#                 else:
#                     print("Invalid face image region, skipping...")

# def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
#     global frame
#     if result.detections:
#         for detection in result.detections:
#             category = detection.categories[0]
#             confidence = category.score
#             print(f"Detection confidence: {confidence}")

#             # 如果置信度足够高，则提取人脸区域
#             if confidence > 0.65:
#                 bbox = detection.bounding_box
#                 x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

#                 # 转换图像格式为 OpenCV 格式
#                 frame = np.array(output_image.numpy_view())

#                 # 确保裁剪区域在图像范围内
#                 x = max(0, x)
#                 y = max(0, y)
#                 w = min(w, frame.shape[1] - x)
#                 h = min(h, frame.shape[0] - y)

#                 face_image = frame[y:y+h, x:x+w]

#                 # 可视化裁剪的人脸区域
#                 if face_image.size > 0:
#                     # 调用表情识别函数对裁剪的人脸进行表情预测
#                     probabilities = predict_emotion(face_image, model, device)
#                     selected_probabilities = format_emotion_result(probabilities)
#                     print(f"Emotion prediction: {selected_probabilities}")

#                     # **绘制检测框和表情预测结果**
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绘制绿色矩形框
#                     emotion_text = ", ".join([f"{key}: {value:.2f}" for key, value in selected_probabilities.items()])
#                     cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#                 else:
#                     print("Invalid face image region, skipping...")
def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    global detections
    detections = result.detections  # 保存检测结果到全局变量

    # 逻辑现在被从回掉函数中抽出了
    # if result.detections:
    #     for detection in result.detections:
    #         category = detection.categories[0]
    #         confidence = category.score
    #         print(f"Detection confidence: {confidence}")

    #         if confidence > 0.65:
    #             bbox = detection.bounding_box
    #             x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

    #             # 确保裁剪区域在图像范围内
    #             x, y = max(0, x), max(0, y)
    #             w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

    #             # # 绘制检测框到全局 frame
    #             # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #             # emotion_text = "Detected"  # 简化，假定检测结果为 "Detected"
    #             # cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #             print(f"BBox: x={x}, y={y}, w={w}, h={h}")




# 初始化表情识别模型
model_path = '/home/yufeng/mdk/bin/affecnet7_epoch19_acc0.671.pth'
model, device = load_emotion_model(model_path)

# 创建人脸检测器实例，使用 LIVE_STREAM 模式
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='/home/yufeng/mdk/bin/trained_model/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# with FaceDetector.create_from_options(options) as detector:
#     # 打开默认摄像头
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # 将 OpenCV 图像转换为 MediaPipe 的 Image 对象
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#         frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
#         # 进行异步人脸检测
#         detector.detect_async(mp_image, frame_timestamp_ms)

#         # 在这里展示画面（包含检测结果）
#         cv2.imshow("Emotion Detection", frame)

#         # 按下 'q' 键退出
#         if cv2.waitKey(1) == ord('q'):
#             break

#     # 释放摄像头资源
#     cap.release()
#     cv2.destroyAllWindows()

# 初始化全局变量

def draw_result(frame, sorted_probabilities, x, y, w, h):
    # 绘制矩形框
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色边框

    # 绘制文字（可选）
    if sorted_probabilities:
        top_emotion = max(sorted_probabilities, key=sorted_probabilities.get)
        cv2.putText(frame, f"{top_emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

# 加载模型和设置全局变量
emotion_model, device = load_emotion_model('/home/yufeng/mdk/bin/affecnet7_epoch19_acc0.671.pth')
detections = []

# 初始化 MediaPipe 检测器和摄像头
with FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)  # 使用默认摄像头

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera. Exiting...")
            break

        # 转换图像为 MediaPipe 格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.copy())
        frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # 执行异步检测
        detector.detect_async(mp_image, frame_timestamp_ms)

        # 如果检测到人脸
        if len(detections) > 0:
            # 找到置信度最高的人脸检测框
            best_detection = max(detections, key=lambda d: d.categories[0].score)
            bbox = best_detection.bounding_box
            x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
            x, y = max(0, x), max(0, y)
            w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)

            # 裁剪人脸图像
            face_image = frame[y:y + h, x:x + w]

            # 检查裁剪是否有效
            if face_image is not None and face_image.size > 0:
                # 预处理人脸图像并进行表情识别
                face_image = preprocess_image(face_image)
                probabilities, _ = predict_emotion(face_image, emotion_model, device)

                # 格式化表情结果
                sorted_probabilities = format_emotion_result(probabilities)

                # 显示情绪颜色窗口
                display_emotion_color(sorted_probabilities)

                # 在主帧中绘制检测结果
                frame = draw_result(frame, sorted_probabilities, x, y, w, h)

        # 显示摄像头画面
        cv2.imshow('Emotion Detection', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program...")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()




# 作为输入部分的显示，即输入表情识别模型的部分


# detections = []
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

#         # 遍历所有检测到的人脸
#         for i, detection in enumerate(detections):  # 使用索引区分多张人脸
#             bbox = detection.bounding_box
#             x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

#             # 确保裁剪区域在图像范围内
#             x, y = max(0, x), max(0, y)
#             w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)

#             # 提取人脸区域
#             face_image = frame[y:y+h, x:x+w]

#             # 检查裁剪的人脸区域是否有效
#             if face_image.size > 0:
#                 # 显示裁剪的人脸区域
#                 window_name = f"Face {i}"  # 用窗口名区分多张人脸
#                 cv2.imshow(window_name, face_image)

#         # 按 'q' 键退出
#         if cv2.waitKey(1) == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# ====================================================================================================================================

