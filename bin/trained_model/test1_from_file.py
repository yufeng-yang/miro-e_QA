import os
import torch
from extract_integrated_feature_tools import extract_fused_feature
from extract_audio_feature_tools import initionalize_audio_model
from extract_video_feature_tools import initialize_face_detector, load_emotion_model
import csv

audio_model = initionalize_audio_model()
mediapipe_model = initialize_face_detector()
emotion_model = load_emotion_model()

def extract_label_from_filename(filename):
    """
    从 RAVDESS 文件名中提取情绪标签。
    Args:
        filename (str): 文件名，例如 "02-01-06-01-02-01-12.mp4"
    Returns:
        int: 对应的情绪标签，例如 6
    """
    # 按分隔符拆分文件名
    parts = filename.split('-')
    # 第三部分为情绪标签，去掉前导零并转换为整数
    emotion_label = int(parts[2])
    return emotion_label

# def process_dataset(dataset_path, save_path, audio_model, mediapipe_model, emotion_model, frame_interval=2):
#     """
#     处理整个数据集，将特征和情绪标签存储在一个文件中。
#     跳过不需要的情绪类别。
#     Args:
#         dataset_path (str): 包含视频文件的数据集路径。
#         save_path (str): 保存融合特征及标签的文件路径。
#         audio_model, mediapipe_model, emotion_model: 预加载的模型。
#         frame_interval (int): 视频帧提取间隔。
#     """
#     data = []
#     skip_emotion = {2}  # 设置需要跳过的情绪类别，例如 Calm = 2
#     for filename in os.listdir(dataset_path):
#         if filename.endswith('.mp4'):
#             video_path = os.path.join(dataset_path, filename)
#             # 提取情绪标签
#             label = extract_label_from_filename(filename)
#             if label in skip_emotion:
#                 print(f"跳过文件：{filename}（情绪标签：{label}）")
#                 continue  # 跳过不需要的情绪类别
#             # 提取融合特征
#             fused_feature = extract_fused_feature(
#                 video_path, audio_model, mediapipe_model, emotion_model, frame_interval=frame_interval
#             )
#             # 存储特征和标签
#             data.append({'features': fused_feature, 'label': label})
    
#     # 保存特征及标签
#     torch.save(data, save_path)
#     print(f"数据集处理完成，已保存到 {save_path}")

# # 示例调用
# dataset_path = "C:\\Users\\28402\\Downloads\\Video_Speech_Actor_01"
# save_path = "filtered_fused_features.pt"
# process_dataset(dataset_path, save_path, audio_model, mediapipe_model, emotion_model, frame_interval=2)


def process_dataset_to_csv(dataset_path, csv_path, audio_model, mediapipe_model, emotion_model, frame_interval=2):
    """
    处理整个数据集，将特征和情绪标签存储为 CSV 文件。
    跳过不需要的情绪类别。
    Args:
        dataset_path (str): 包含视频文件的数据集路径。
        csv_path (str): 保存融合特征及标签的 CSV 文件路径。
        audio_model, mediapipe_model, emotion_model: 预加载的模型。
        frame_interval (int): 视频帧提取间隔。
    """
    skip_emotion = {2}  # 跳过 Calm 的情绪标签
    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入 CSV 表头
        feature_length = 512 + 768  # Video feature (512) + Audio feature (768)
        header = ['Label'] + [f'Feature_{i}' for i in range(feature_length)]
        csv_writer.writerow(header)

        # 遍历数据集并提取特征和标签
        for filename in os.listdir(dataset_path):
            if filename.endswith('.mp4'):
                video_path = os.path.join(dataset_path, filename)
                label = extract_label_from_filename(filename)
                if label in skip_emotion:
                    print(f"跳过文件：{filename}（情绪标签：{label}）")
                    continue
                fused_feature = extract_fused_feature(
                    video_path, audio_model, mediapipe_model, emotion_model, frame_interval=frame_interval
                )
                # 将特征和标签写入 CSV
                row = [label] + fused_feature.squeeze(0).tolist()
                csv_writer.writerow(row)

        print(f"数据集处理完成，已保存到 {csv_path}")

# 示例调用
dataset_path = "C:\\Users\\28402\\Downloads\\Video_Speech_Actor_01\\actor_01"
csv_path = "fused_features.csv"
process_dataset_to_csv(dataset_path, csv_path, audio_model, mediapipe_model, emotion_model, frame_interval=2)