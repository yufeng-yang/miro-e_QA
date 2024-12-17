import torch
import torch.nn as nn
import torch.nn.functional as F
from extract_integrated_feature_tools import extract_fused_feature, extract_solo_audio_feature, extract_solo_video_feature, extract_audio_from_video
from extract_audio_feature_tools import initionalize_audio_model, initionalize_large_audio_model, extract_audio_from_video
from extract_video_feature_tools import initialize_face_detector, load_emotion_model, feature_to_7probability, video_to_frames, extract_face_area_frame, image_input_preprocessing, extrace_dealt_feature

class EmotionClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),  # First hidden layer
            nn.ReLU(),
            nn.BatchNorm1d(512),         # Add BatchNorm
            nn.Linear(512, 256),         # Second hidden layer
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),         # Third hidden layer
            nn.ReLU(),
            nn.Dropout(0.05),            # Lower Dropout
            nn.Linear(128, num_classes)  # Output layer
        )

    def forward(self, x):
        return self.model(x)

class EmotionRecognitionPipeline:
    def __init__(self, classifier_model_path):
        # Load models
        self.audio_model = initionalize_audio_model()
        self.mediapipe_model = initialize_face_detector()
        self.emotion_model = load_emotion_model()
        
        # Load trained emotion classifier model
        input_size = 1280  # Video + Audio fused feature size
        num_classes = 7     # Number of emotion classes
        self.classifier_model = EmotionClassifier(input_size, num_classes)
        self.classifier_model.load_state_dict(torch.load(classifier_model_path))
        self.classifier_model.eval()  # Set to evaluation mode

    def extract_1280_features(self, video_path, frame_interval=2, save_path='./fused_feature.pt'):
        # Extract fused features from video and audio
        fused_feature = extract_fused_feature(video_path, self.audio_model, self.mediapipe_model, self.emotion_model, frame_interval, save_path)
        return fused_feature
    
    # 我们的过程似乎是要分别提起音频和视频特征，然后融合特征
    def extract_512_video_features(self, video_path, frame_interval=2):
        # Extract video features
        video_feature = extract_solo_video_feature(self.mediapipe_model, self.emotion_model, video_path, frame_interval)
        return video_feature
    
    def extract_768_audio_features(self, audio_path):
        # Extract audio features
        audio_feature, audio_full = extract_solo_audio_feature(self.audio_model, audio_path)
        return audio_feature, audio_full
    def predict_emotion(self, fused_feature):
        # Ensure feature dimension is correct
        if fused_feature.dim() == 1:
            fused_feature = fused_feature.unsqueeze(0)  # Add batch dimension
        
        # Predict emotion probabilities
        with torch.no_grad():
            output = self.classifier_model(fused_feature)
            probabilities = F.softmax(output, dim=1)  # Calculate probabilities for each class
        
        # Map probabilities to emotion labels
        label_mapping = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Angry", 4: "Fearful", 5: "Disgust", 6: "Surprised"}
        probability_dict = {label_mapping[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        
        return probability_dict

    def get_dominant_emotion(self, probability_dict):
        # Get the emotion with the highest probability
        dominant_emotion = max(probability_dict, key=probability_dict.get)
        return dominant_emotion

import numpy as np

def combine_emotion_results(video_probs, video_labels, audio_scores, audio_labels, weights=(0.5, 0.5)):
    """
    将 video 和 audio 的情感分析结果结合。

    Args:
        video_probs (torch.Tensor): Video 模型的概率分布 (1, 7)。
        video_labels (list): Video 模型的标签。
        audio_scores (list): Audio 模型的得分分布。
        audio_labels (list): Audio 模型的标签。
        weights (tuple): Video 和 Audio 权重，默认均为 0.5。

    Returns:
        dict: 结合后的情感标签和概率分布。
    """
    # 确保 video_probs 是 NumPy 数组
    video_probs = video_probs.numpy().flatten()

    # 建立标签映射
    class7_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']
    audio_to_video_mapping = {
        '生气/angry': 'Angry',
        '厌恶/disgusted': 'Disgust',
        '恐惧/fearful': 'Fear',
        '开心/happy': 'Happy',
        '中立/neutral': 'Neutral',
        '难过/sad': 'Sad',
        '吃惊/surprised': 'Surprise'
    }

    # 将 audio_scores 转换到 video 的标签顺序
    audio_probs = np.zeros(len(class7_names))
    for i, label in enumerate(audio_labels):
        if label in audio_to_video_mapping:
            video_index = class7_names.index(audio_to_video_mapping[label])
            audio_probs[video_index] = audio_scores[i]

    # 归一化 audio_probs
    audio_probs /= np.sum(audio_probs)

    # 加权融合
    w1, w2 = weights
    final_probs = w1 * video_probs + w2 * audio_probs

    # 构建最终结果
    result = {label: prob for label, prob in zip(class7_names, final_probs)}
    return result


def main(video_path,audio_path):
    # Initialize the emotion recognition pipeline
    classifier_model_path = '/home/yufeng/mdk/emotion_classifier_all_best.pth'
    pipeline = EmotionRecognitionPipeline(classifier_model_path)
    
    # Extract features from the given video
    # fused_feature = pipeline.extract_1280_features(video_path)

    # 尝试一下分开提取再融合的结果
    video_feature = pipeline.extract_512_video_features(video_path)
    # audio_path = extract_audio_from_video(video_path)
    # 实际操作中，不需要从video中抓取audio，因为audio已经准备好了
    audio_feature, audio_full = pipeline.extract_768_audio_features(audio_path)
    fused_feature = torch.cat([video_feature, audio_feature], dim=-1)  # Shape: [1, 1280]
    
    # Predict emotion
    probability_dict = pipeline.predict_emotion(fused_feature)
    
    # Get the dominant emotion
    dominant_emotion = pipeline.get_dominant_emotion(probability_dict)
    
    # Print emotion probabilities
    print("Emotion Probabilities:")
    for emotion, prob in probability_dict.items():
        print(f"{emotion}: {prob:.4f}")
    
    # Based on the dominant emotion, trigger specific actions
    if dominant_emotion == "Happy":
        print("Detected Happy emotion. Executing happy feedback actions...")
        # Here you would call your HappyMoodFeedback class to execute the actions
        # Example:
        # interface = miro.lib.RobotInterface()
        # controller = HappyMoodFeedback(interface)
        # execute_all_functions(controller, duration=9)


def separate_main(video_path, audio_path,mediapipe_model,emotion_model):
    # 这个函数的目的是完全不管刚刚定义的类

    # ===================== Video 部分 =====================
    # 初始化视频部分的两个模型


    # 准备好 video 部分的统计变量
    feature_count = 0
    accumulated_feature = None

    # 开始提取特征
    frames_dataset = video_to_frames(video_path, frame_interval=5)
    for frame in frames_dataset:
        face_image = extract_face_area_frame(mediapipe_model, frame)
        input_image = image_input_preprocessing(face_image)
        _, dealt_feature = extrace_dealt_feature(emotion_model, input_image)
        if face_image:
            if accumulated_feature is None:
                accumulated_feature = dealt_feature
            else:
                accumulated_feature += dealt_feature
            feature_count += 1

    print(f"目前共 {feature_count} 次出现人脸")
    if accumulated_feature is not None:
        accumulated_feature /= feature_count
        video_probabilities = feature_to_7probability(emotion_model, accumulated_feature)
        print("Video Probabilities:", video_probabilities)
    else:
        video_probabilities = None

    # ===================== Audio 部分 =====================
    audio_model = initionalize_large_audio_model()
    audio_feature, audio_full = extract_solo_audio_feature(audio_model, audio_path)
    print("Audio Full Result:", audio_full)

    # 提取 audio 的标签和得分
    audio_labels = audio_full['labels']
    audio_scores = audio_full['scores']
    # 将 audio_scores 转换为 NumPy 数组并加权
    audio_scores = np.array(audio_scores) * 0.5

    # ===================== 融合部分 =====================
    if video_probabilities is not None:
        final_result = combine_emotion_results(
            video_probabilities,  # Video 的概率分布
            ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry'],  # Video 的标签
            audio_scores,  # Audio 的分数
            audio_labels  # Audio 的标签
        )
        # 找到概率最高的标签
        highest_label = max(final_result, key=final_result.get)
        highest_prob = final_result[highest_label]
        print("最终融合结果:")
        for label, prob in final_result.items():
            print(f"{label}: {prob:.4f}")
    else:
        print("没有检测到人脸，直接返回 Audio 结果。")
    
    # 返回最终结果
    return highest_label, highest_prob

def separate_frame_main(video_average_feature, audio_path):
    '''
    这里这个方法是直接
    '''



if __name__ == "__main__":
    mediapipe_model = initialize_face_detector()
    emotion_model = load_emotion_model()
    video_path = '/home/yufeng/mdk/user_video.avi'  # Example video path
    # main(video_path,None)
    audio_path = '/home/yufeng/mdk/user_input.wav'
    separate_main(video_path, audio_path,mediapipe_model,emotion_model)


