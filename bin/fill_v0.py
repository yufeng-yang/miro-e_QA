import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time

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
                        class_names.index('Angry')]
    selected_probabilities = {class_names[i]: probabilities[i] for i in selected_indices}
    return selected_probabilities

# 回调函数，用于处理检测结果。这是LIVE模式下mediapipe必备的
def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    if result.detections:
        for detection in result.detections:
            category = detection.categories[0]
            confidence = category.score
            print(f"Detection confidence: {confidence}")

            # 如果置信度足够高，则提取人脸区域
            if confidence > 0.65:
                bbox = detection.bounding_box
                x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

                # 转换图像格式为 OpenCV 格式
                frame = np.array(output_image.numpy_view())

                # 确保裁剪区域在图像范围内
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                face_image = frame[y:y+h, x:x+w]

                # 可视化裁剪的人脸区域
                if face_image.size > 0:

                    # 调用表情识别函数对裁剪的人脸进行表情预测
                    probabilities = predict_emotion(face_image, model, device)
                    selected_probabilities = format_emotion_result(probabilities)
                    print(f"Emotion prediction: {selected_probabilities}")
                else:
                    print("Invalid face image region, skipping...")

# 初始化表情识别模型
model_path = 'C:\\CodeSpace\\Integration and voting\\affecnet7_epoch19_acc0.671.pth'
model, device = load_emotion_model(model_path)

# 创建人脸检测器实例，使用 LIVE_STREAM 模式
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='C:\\CodeSpace\\google_face_detector\\blaze_face_short_range (1).tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with FaceDetector.create_from_options(options) as detector:
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将 OpenCV 图像转换为 MediaPipe 的 Image 对象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # 进行异步人脸检测
        detector.detect_async(mp_image, frame_timestamp_ms)

        # 按下 'q' 键退出
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()
