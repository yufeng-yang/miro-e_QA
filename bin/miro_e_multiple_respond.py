from Listening_to_generate_respond import generate_single_respond
from temp_audio_play import AudioPlayback
import miro2 as miro
import rospy

# 初始化MIRO机器人接口
interface = miro.lib.RobotInterface()

# 多次监听的循环
while not rospy.is_shutdown():
    sound_file = generate_single_respond()
    if sound_file is not None:
        playback = AudioPlayback(interface, "/miro", sound_file)
        rospy.sleep(2)
        playback.play()