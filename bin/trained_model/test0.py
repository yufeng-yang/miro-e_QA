from extract_integrated_feature_tools import extract_fused_feature
from extract_audio_feature_tools import initionalize_audio_model
from extract_video_feature_tools import initialize_face_detector, load_emotion_model
import torch

# Load models
audio_model = initionalize_audio_model()
mediapipe_model = initialize_face_detector()
emotion_model = load_emotion_model()

# Extract fused feature
video_path = 'C:\\CodeSpace\\Integration and voting\\test_source\\01-01-05-02-02-02-01.mp4'
fused_feature1 = extract_fused_feature(video_path, audio_model, mediapipe_model, emotion_model, frame_interval=2, save_path='./f1.pt')
