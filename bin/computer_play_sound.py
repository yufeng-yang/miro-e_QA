import pyaudio
import wave

def play_sound(file_path):
    """
    播放指定路径的 WAV 文件。
    
    参数:
        file_path (str): WAV 文件的路径。
    """
    try:
        # 打开音频文件
        wf = wave.open(file_path, 'rb')
        
        # 初始化 PyAudio
        audio = pyaudio.PyAudio()

        # 设置音频流参数
        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

        # 读取音频数据并播放
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        # 停止并关闭流
        stream.stop_stream()
        stream.close()

        # 关闭 PyAudio
        audio.terminate()
        wf.close()

        print("音频播放完成！")

    except Exception as e:
        print(f"播放音频时发生错误: {e}")