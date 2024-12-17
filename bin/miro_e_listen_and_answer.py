from Listening_to_generate_respond import generate_single_respond

from temp_audio_play import AudioPlayback

import miro2 as miro
import rospy

interface = miro.lib.RobotInterface() 

sound_file = generate_single_respond()
playback = AudioPlayback(interface, "/miro", sound_file)
rospy.sleep(2)
playback.play()