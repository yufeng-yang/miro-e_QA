'''
Using the emotion representation model
rec_result only contains {'feats'}
	granularity="utterance": {'feats': [*768]}
	granularity="frame": {feats: [T*768]}
'''
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import ffmpeg

def extract_audio_from_video(video_path, output_audio_path = './audio.wav'):
    # 提取音频, 自动覆盖
    ffmpeg.input(video_path).output(output_audio_path).overwrite_output().run()
    # print(f"Audio extracted and saved to {output_audio_path}")
    return output_audio_path

def initionalize_audio_model():
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_plus_base")
    return inference_pipeline

def extract_audio_feature(audio_model, audio_path, granularity="utterance", extract_embedding=True):
    rec_result = audio_model(audio_path, granularity=granularity, extract_embedding=extract_embedding)
    # return rec_result[0]['key'], rec_result[0]['labels'], rec_result[0]['scores'], rec_result[0]['feats']
    # 算了，我想了想还是把一整个字典结构output出去吧
    return rec_result


# audio_model = initionalize_audio_model()
# audio = 'C:\\CodeSpace\\Integration and voting\\test_source\\03-01-05-02-01-01-01.wav'
# outp = extract_audio_feature(audio_model,audio)
# print(outp)