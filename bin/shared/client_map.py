#!/usr/bin/python3
#
#	@section COPYRIGHT
#	Copyright (C) 2023 Consequential Robotics Ltd
#	
#	@section AUTHOR
#	Consequential Robotics http://consequentialrobotics.com
#	
#	@section LICENSE
#	For a full copy of the license agreement, and a complete
#	definition of "The Software", see LICENSE in the MDK root
#	directory.
#	
#	Subject to the terms of this Agreement, Consequential
#	Robotics grants to you a limited, non-exclusive, non-
#	transferable license, without right to sub-license, to use
#	"The Software" in accordance with this Agreement and any
#	other written agreement with Consequential Robotics.
#	Consequential Robotics does not transfer the title of "The
#	Software" to you; the license granted to you is not a sale.
#	This agreement is a binding legal agreement between
#	Consequential Robotics and the purchasers or users of "The
#	Software".
#	
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
#	KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#	WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
#	OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
#	OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#	SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#	

import rospy
import miro2 as miro
import time
import sys
import os
import numpy as np

################################################################

def error(msg):

	print(msg)
	sys.exit(0)

def fmt(msg):

	x = msg.data
	s = "{0:.3f} {1:.3f}".format(x[0], x[1])
	return s

################################################################

class client_map:

	def callback_sensors(self, msg):
	
		# get robot feedback
		spd = np.array(msg.wheel_speed_opto.data)
		kin = np.array(msg.kinematic_joints.position)
		
		# integrate speed to estimate position
		T = 0.02
		twist = msg.odom.twist.twist
		dr = twist.linear.x
		dtheta = twist.angular.z
		self.pose[2] += dtheta * T
		dxy = miro.lib.kc.kc_rotate(np.array([dr, 0.0, 0.0]), 'z', self.pose[2])
		self.pose += [dxy[0], dxy[1], 0.0]
		
		# update kc
		self.kc.setConfig(kin)
		
		# demonstrate mapping from camera image space to FOOT
		# using a pixel near the middle of a camera image
		p = [160, 90]
		
		# of the zeroth (left) camera
		stream_index = 0
		
		# and a presumed range
		r = 0.5
		
		# map to a view line in CAM
		v = self.cam.p2v(p)

		# map to a location in HEAD
		oh = self.cam.v2oh(stream_index, v, r)
		
		# and use kc to map that into FOOT
		of = self.kc.changeFrameAbs(miro.constants.LINK_HEAD, miro.constants.LINK_FOOT, oh)
		
		# finally, if we'd been integrating the speed so that we had a
		# current estimate of the robot position, we could...
		# ...oh, wait, we have!
		self.kc.setPose(self.pose)
		
		# so we can, finally, map the target into WORLD
		ow = self.kc.changeFrameAbs(miro.constants.LINK_HEAD, miro.constants.LINK_WORLD, oh)
		
		# report
		print ("target in WORLD", ow)
		
	def loop(self):

		# loop
		while not rospy.core.is_shutdown():

			# state
			time.sleep(0.1)

	def __init__(self):

		# state
		self.cam = miro.lib.camera_model.CameraModel()
		self.kc = miro.lib.kc_interf.kc_miro()
		self.pose = np.array([0.0, 0.0, 0.0])

		# this needs to be set based on the actual frame size, which
		# can be obtained from the camera frame topic. here, we just
		# assume the default frame size is in use.
		self.cam.set_frame_size(320, 180)

		# robot name
		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		# subscribe
		topic = topic_base_name + "/sensors/package"
		print ("subscribe", topic)
		self.sub_log = rospy.Subscriber(topic, miro.msg.sensors_package, self.callback_sensors, queue_size=5, tcp_nodelay=True)

if __name__ == "__main__":

	rospy.init_node("client_map", anonymous=True)
	main = client_map()
	main.loop()




