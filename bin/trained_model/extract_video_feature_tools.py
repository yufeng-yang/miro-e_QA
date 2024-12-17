import cv2
import mediapipe as mp
from PIL import Image
import torch
from networks.DDAM import DDAMNet
import torchvision.transforms as transforms
# 1.首先是从视频到frame的提取
def video_to_frames(video_path, frame_interval=1):
    """
    从视频中提取每隔一定间隔的帧并返回这些帧。

    参数:
        video_path (str): 输入视频文件的路径。
        frame_interval (int): 采样间隔，默认每隔1帧返回一张图片。

    返回:
        frames (list): 一个包含提取帧的列表，每个元素都是一个图像数组。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0  # 当前帧计数
    
    while cap.isOpened():
        ret, frame = cap.read()  # 读取一帧
        if not ret:
            break  # 如果没有帧可读，退出循环

        # 每隔 frame_interval 帧返回一张图片
        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    print(f"目前设置为每 {frame_interval}提取一次，共提取了 {len(frames)} 张帧")
    return frames


# 2.然后是初始化mediapipe模型
def initialize_face_detector(model_path='C:\\CodeSpace\\integrate_video_audio_features\\networks\\blaze_face_short_range.tflite'):
    """
    初始化 FaceDetector 实例。
    
    参数:
        model_path (str): tflite 模型的路径。
        
    返回:
        detector (FaceDetector): 初始化的人脸检测器实例。
    """
    # 配置 mediapipe FaceDetector 选项
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )
    mediapipe_detector = FaceDetector.create_from_options(options)
    return mediapipe_detector

# 3.然后是从frame中提取头部区域
# 这是用作frame作为输入来裁剪头部区域的函数
def extract_face_area_frame(detector, frame):
    """
    从给定的视频帧中提取人脸区域。
    
    参数:
        detector (FaceDetector): 已初始化的人脸检测器实例。
        frame (np.ndarray): 视频帧，格式为 OpenCV 读取的 BGR 格式。
        
    返回:
        cropped_face_image (PIL.Image): 裁剪出的人脸区域图像。如果没有检测到人脸，返回 None。
    """
    # 将 OpenCV 的 BGR 图像转换为 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 将 NumPy 数组转换为 mediapipe 图像格式
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # 人脸检测
    face_detector_result = detector.detect(mp_image)
    
    # 如果检测到人脸，提取并裁剪人脸区域
    if face_detector_result.detections:
        # 获取边界框
        detection = face_detector_result.detections[0]  # 假设只处理第一张人脸
        bbox = detection.bounding_box
        # print("Bounding box:", bbox)
        
        # 将 mp.Image 转换为 numpy 数组以便裁剪
        image_np = mp_image.numpy_view()
        
        # 裁剪人脸区域
        cropped_face = image_np[bbox.origin_y:bbox.origin_y + bbox.height, bbox.origin_x:bbox.origin_x + bbox.width]
        
        # 将裁剪后的图像转换为 PIL 图像
        cropped_face_image = Image.fromarray(cropped_face)
        return cropped_face_image
    else:
        print("No face detected.")
        return None
    

# 4.用作单个图像输入来裁剪头部区域的函数
def extract_face_area_image(detector, image_path):
    """
    从给定图像中提取人脸区域。
    
    参数:
        detector (FaceDetector): 已初始化的人脸检测器实例。
        image_path (str): 图像文件的路径。
        
    返回:
        cropped_face_image (PIL.Image): 裁剪出的人脸区域图像。如果没有检测到人脸，返回 None。
    """
    # 加载图像
    mp_image = mp.Image.create_from_file(image_path)
    
    # 人脸检测
    face_detector_result = detector.detect(mp_image)
    
    # 如果检测到人脸，提取并裁剪人脸区域
    if face_detector_result.detections:
        # 获取边界框
        detection = face_detector_result.detections[0]  # 只处理第一张人脸
        bbox = detection.bounding_box
        print("Bounding box:", bbox)
        
        # 将 mp.Image 转换为 numpy 数组以便裁剪
        image_np = mp_image.numpy_view()
        
        # 裁剪人脸区域
        cropped_face = image_np[bbox.origin_y:bbox.origin_y + bbox.height, bbox.origin_x:bbox.origin_x + bbox.width]
        
        # 将裁剪后的图像转换为 PIL 图像
        cropped_face_image = Image.fromarray(cropped_face)
        return cropped_face_image
    else:
        print("No face detected.")
        return None

# 5. 加载表情识别的模型
def load_emotion_model():
    # 选定设备
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化模型
    model = DDAMNet(num_class=7, num_head=2, pretrained=True)
    # 这里是将模型加载到GPU上（选定的device）
    # model = model.to(device)

    
    # 加载模型权重
    checkpoint = torch.load('C:\\CodeSpace\\integrate_video_audio_features\\networks\\affecnet7_epoch19_acc0.671.pth')
    # print("Keys in checkpoint:", list(checkpoint.keys()))
    # Keys in checkpoint: ['iter', 'model_state_dict', 'optimizer_state_dict'],这里可以看出。加载的权重文件是一个字典型。
    # 其中，iter代表迭代次数，model_state_dict代表权重，optimizer顾名思义，就是优化器参数，例如学习率，优化器等
    # 所以需要使用key访问到权重并加载。
    model.load_state_dict(checkpoint['model_state_dict'])
    # 切换模型到评估模式，即避免训练时才会有的例如0输出的dropout层等。
    model.eval()
    return model

# 6. 表情图像的预处理
# 下来自然是要将图片转化成符合神经网络的输入格式，查看得知，要求的是3*112*112的输入
def image_input_preprocessing(img):
    # 选定设备,既然模型加载到了gpu上，那图像就也加载上来。
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载并预处理图像
    # 检查是否已经是 PIL 图像，如果不是则加载
    if not isinstance(img, Image.Image):
        img = Image.open(img).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((112,112)),  # 模型需要 3*112*112 输入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化,问就是推荐
    ])
    # 增加批量维度，即会将原本的3*224*224 变成1*3*224*224
    input_data = transform(img).unsqueeze(0)  
    # 移动设备
    # input_data = input_data.to(device) 
    return input_data

# 7. 调用表情识别模型,对于单个frame或者单张图片
# 最后一步，即使用模型了
def extrace_dealt_feature(model, input_data):
    # 使用 torch.no_grad() 禁用梯度计算
    with torch.no_grad():
        # 这里需要注意的一点是：model()实际上是在调用forward方法。
        output, _, _, y = model(input_data, return_dealt_feature = True)
    return output, y

# 8. 调用表情识别模型,对于多个frame
# 由于我们需要对多个帧进行平均，所以我们需要将所有帧的特征相加，然后除以帧数以获得平均特征。
# 施工中...

# 9. 将特征转换为概率分布
# 将特征 y 输入到模型的 fc 层，获得最终的概率分布,这里直接把softmax也做了
# fc曾就是fully connect层，它是将原本512大小的feature变成7大小，每一个代表一种emotion的概率
def feature_to_7probability(model, feature):
    # 使用模型的 fc 层进行分类，并应用 softmax 转为概率分布
    with torch.no_grad():
        logits = model.fc(feature)  # 调用模型中的 fc 层，这个fc曾也是被训练过了
        probabilities = torch.nn.functional.softmax(logits, dim=1)  # 转为概率分布
    return probabilities

# # 测试代码：=======================================================

# # # 首先加载两个模型
# mediapipe_model = initialize_face_detector()
# emotion_model = load_emotion_model()

# # 准备一些用来平均的东西
# accumulated_feature = None
# feature_count = 0
# # 然后是使用他们
# # video_path = 'C:\\CodeSpace\\Integration and voting\\test_source\\01-01-02-01-01-01-01.mp4'
# # video_path = 'C:\\CodeSpace\\Integration and voting\\test_source\\01-01-05-02-02-02-01.mp4'
# video_path = 'C:\\CodeSpace\\Integration and voting\\test_source\\01-01-03-02-01-01-01.mp4'
# frames_dataset = video_to_frames(video_path, frame_interval=10)
# for frame in frames_dataset:
#     face_image = extract_face_area_frame(mediapipe_model, frame)
#     input_image = image_input_preprocessing(face_image)
#     # print(input_image.shape)

#     # # !!!!!测试1: 分帧输出
#     # _, dealt_feature = extrace_dealt_feature(emotion_model, input_image)
#     # final_probabilities = feature_to_7probability(emotion_model, dealt_feature)
#     # print(final_probabilities)

#     # 测试2: 平均输出

#     _, dealt_feature = extrace_dealt_feature(emotion_model, input_image)
#     if face_image:
#         if accumulated_feature is None:
#             accumulated_feature = dealt_feature
#         else:
#             accumulated_feature += dealt_feature
#         feature_count += 1
# print(f"目前共{feature_count}次出现人脸")
# if accumulated_feature is not None:
#     accumulated_feature /= feature_count
#     final_probabilities = feature_to_7probability(emotion_model, accumulated_feature)
#     print(final_probabilities)
# # ===============================================================
