import speech_recognition as sr

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("请说话...")
        # 降低环境噪音
        recognizer.adjust_for_ambient_noise(source)
        # 这个audio对象是一个AudioData对象，包含了从麦克风中录取的音频数据
        audio = recognizer.listen(source)

        audio_path = "captured_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio.get_wav_data())

    try:
        print("recognizing and judging your mood...")
        text = recognizer.recognize_google(audio, language="en-US")
        print(f"(your mood is UNDEFINED) you said: {text}")
        return text
    except sr.UnknownValueError:
        print("抱歉，无法理解你的话。")
        return None
    except sr.RequestError:
        print("无法连接到语音识别服务。")
        return None
    
if __name__ == "__main__":
    recognize_speech_from_mic()