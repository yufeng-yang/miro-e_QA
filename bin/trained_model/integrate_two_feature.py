from extract_audio_feature_tools import initionalize_audio_model, extract_audio_feature, extract_audio_from_video
full_file_path = 'C:\\CodeSpace\\Integration and voting\\test_source\\01-01-05-02-02-02-01.mp4'

# 首先提取audio的特征

audio_path = extract_audio_from_video(full_file_path)
audio_model = initionalize_audio_model()
audio_output = extract_audio_feature(audio_model, audio_path)

# 然后提取video的特征
from extract_video_feature_tools import initialize_face_detector, load_emotion_model, video_to_frames, extract_face_area_frame, image_input_preprocessing, extrace_dealt_feature, feature_to_7probability
import torch
mediapipe_model = initialize_face_detector()
emotion_model = load_emotion_model()
frames_dataset = video_to_frames(full_file_path, frame_interval=2)
accumulated_feature = None
feature_count = 0
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

# print(accumulated_feature.dtype)
# ===================== Integration =====================
# 最后将两个特征结合起来
# 首先是video的特征
video_feature = accumulated_feature / feature_count
# Shape: [1, 512]
# 归一化video特征
# 平均后对最终特征L2归一化
video_feature = accumulated_feature / feature_count
video_feature = video_feature / torch.norm(video_feature, p=2)

# 其次是audio的特征
audio_feature = torch.tensor(audio_output[0]['feats'], dtype=torch.float32)
# 对 audio_feature 也进行 L2 归一化
audio_feature = audio_feature / torch.norm(audio_feature, p=2)
# Convert audio_feature to torch.Tensor and adjust its shape
audio_feature = torch.tensor(audio_feature).unsqueeze(0)  # Shape: [1, 768]



# Concatenate along the last dimension
fused_feature = torch.cat([video_feature, audio_feature], dim=-1)

torch.save(fused_feature, 'fused_feature.pt')

# ===================== Test =====================

# # Print the result for verification
# print("Fused Feature:")
# print("Type:", type(fused_feature))
# print("Shape:", fused_feature.shape)

# Print details for debugging
# print("Video Feature:")
# print("Type:", type(video_feature))
# print("Shape:", video_feature.shape)

# print("\nAudio Feature:")
# print("Type:", type(audio_feature))
# print("Shape:", audio_feature.shape)

# fused_feature = torch.cat([video_feature, audio_feature], dim=-1)
# print(fused_feature.shape)

# # 检查 MixedFeatureNet 输出的范数
# print(f"Video feature L2 norm: {torch.norm(video_feature, 2, dim=1)}")

# # 检查音频特征的范数
# print(f"Audio feature L2 norm: {torch.norm(audio_feature, 2)}")

# print(f"Final fused feature L2 norm: {torch.norm(fused_feature, 2)}")

