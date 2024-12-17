import torch
from extract_audio_feature_tools import initionalize_audio_model, extract_audio_feature, extract_audio_from_video
from extract_video_feature_tools import initialize_face_detector, load_emotion_model, video_to_frames, extract_face_area_frame, image_input_preprocessing, extrace_dealt_feature, feature_to_7probability

def extract_fused_feature(video_path, audio_model, mediapipe_model, emotion_model, frame_interval=2, save_path=None):
    """
    提取视频的综合音视频特征，并返回或保存为 .pt 文件。

    Args:
        video_path (str): 视频文件路径。
        audio_model: 已初始化的音频模型。
        mediapipe_model: 已初始化的人脸检测模型。
        emotion_model: 已加载的视频情感特征提取模型。
        frame_interval (int, optional): 提取视频帧的间隔，默认每 2 帧提取一次。
        save_path (str, optional): 如果指定路径，则保存提取的特征为 .pt 文件。

    Returns:
        torch.Tensor: 综合特征，形状为 [1, 1280]。
    """
    # 提取音频特征
    audio_path = extract_audio_from_video(video_path)
    audio_output = extract_audio_feature(audio_model, audio_path)
    audio_feature = torch.tensor(audio_output[0]['feats'], dtype=torch.float32)
    audio_feature = audio_feature / torch.norm(audio_feature, p=2)
    audio_feature = audio_feature.unsqueeze(0)  # Shape: [1, 768]

    # 提取视频特征
    frames_dataset = video_to_frames(video_path, frame_interval=frame_interval)
    accumulated_feature = None
    feature_count = 0

    for frame in frames_dataset:
        face_image = extract_face_area_frame(mediapipe_model, frame)
        if face_image is None:
            continue
        input_image = image_input_preprocessing(face_image)
        _, dealt_feature = extrace_dealt_feature(emotion_model, input_image)

        if accumulated_feature is None:
            accumulated_feature = dealt_feature
        else:
            accumulated_feature += dealt_feature
        feature_count += 1

    if feature_count == 0:
        raise ValueError(f"No valid frames with faces found in {video_path}")

    video_feature = accumulated_feature / feature_count
    video_feature = video_feature / torch.norm(video_feature, p=2)  # Shape: [1, 512]

    # 融合特征
    fused_feature = torch.cat([video_feature, audio_feature], dim=-1)  # Shape: [1, 1280]

    # 如果指定保存路径，则保存特征
    if save_path:
        torch.save(fused_feature, save_path)
    
    return fused_feature