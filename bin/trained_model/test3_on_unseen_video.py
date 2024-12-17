from extract_integrated_feature_tools import extract_fused_feature
from extract_audio_feature_tools import initionalize_audio_model
from extract_video_feature_tools import initialize_face_detector, load_emotion_model
import torch

# Load models
audio_model = initionalize_audio_model()
mediapipe_model = initialize_face_detector()
emotion_model = load_emotion_model()

# Extract fused feature on one unseen video
# video_path = 'C:\\CodeSpace\\integrate_video_audio_features\\01-01-03-01-01-01-02.mp4'
# video_path = "C:\\Users\\28402\\Pictures\\Camera Roll\\WIN_20241118_15_02_54_Pro.mp4"
# video_path = 'C:\\CodeSpace\\integrate_video_audio_features\\01-01-05-02-01-02-02.mp4'
video_path = "C:\\Users\\28402\Pictures\\Camera Roll\\WIN_20241118_15_06_03_Pro.mp4"
fused_feature1 = extract_fused_feature(video_path, audio_model, mediapipe_model, emotion_model, frame_interval=2, save_path='./f0000.pt')

import torch
import torch.nn as nn
import torch.nn.functional as F

# 加载训练好的模型结构
class EmotionClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 模型参数
input_size = 1280  # Video + Audio fused feature size
num_classes = 7     # Number of emotion classes
model = EmotionClassifier(input_size, num_classes)

# 加载训练好的模型权重
model.load_state_dict(torch.load('emotion_classifier.pth'))
model.eval()  # 切换到评估模式


# 确保特征维度正确
if fused_feature1.dim() == 1:
    fused_feature1 = fused_feature1.unsqueeze(0)  # 增加 batch 维度
    print(fused_feature1)

# 预测概率分布
with torch.no_grad():
    output = model(fused_feature1)
    probabilities = F.softmax(output, dim=1)  # 计算每个类别的概率分布

# 将概率与情绪类别对应
label_mapping = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Angry", 4: "Fearful", 5: "Disgust", 6: "Surprised"}
probability_dict = {label_mapping[i]: prob.item() for i, prob in enumerate(probabilities[0])}

print("Emotion Probabilities:")
for emotion, prob in probability_dict.items():
    print(f"{emotion}: {prob:.4f}")
