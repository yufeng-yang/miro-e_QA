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
from sensor_msgs.msg import CompressedImage

import time
import sys
import os
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

# import april tags if available
try:
	import apriltag
	print ("April Tags library imported OK")
except:
	print ("April Tags library not available (use pip install apriltag)")



################################################################

def error(msg):
	print(msg)
	sys.exit(0)

################################################################

# if no argument provided
if len(sys.argv) == 1:

	# show usage
	print ("pass one of the following arguments to set the mode:")
	print ("\tshow: show video (eye cameras) as it arrives from platform")
	print ("\trecord: record video to /tmp/client_video_*.avi")
	print ("\noptionally include any of the following arguments:")
	print ("\t--april: detect April Tags and annotate video")
	print ("\t--stitch: stitch stereo images into one image")

	# done
	exit()


################################################################

class client:

	# 这里连着三个call_back函数是用于将ros的图像转化为opencv2能处理的图像
	def callback_cam(self, ros_image, index):
		# 传入两个参数，第一个是ros的图像，第二个是索引，用于区分左右眼

		# silently (ish) handle corrupted JPEG frames
		try:

			# convert compressed ROS image to raw CV image
			image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")

			# store image for display
			# 将转化好的opencv2格式的图像存在self.input_camera[0/1]中,也是用于区分左右眼
			self.input_camera[index] = image

		except CvBridgeError as e:

			# swallow error, silently
			#print(e)
			pass

	def callback_caml(self, ros_image):

		self.callback_cam(ros_image, 0)

	def callback_camr(self, ros_image):

		self.callback_cam(ros_image, 1)

	def detect_april(self, im):
		# 这个是用作tag的方法。
		# get grey frame
		im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		# get all detected tags
		detected_tags = self.april_detector.detect(im_grey)

		# for each
		for tag in detected_tags:

			# accept only exact match
			if tag.hamming > 0:
				continue

			# annotate image
			corners = np.array(tag.corners).astype('int32')
			a = corners[3]
			for i in range(4):
				b = corners[i]
				cv2.line(im, (a[0], a[1]), (b[0], b[1]), (255, 0, 255), 2)
				a = b

		# ok
		return im

	def loop(self):

		# state
		# 选择要处理的是左右两个摄像头还是合成的那一个摄像头
		channels_to_process = [0, 1]
		if not self.image_stitcher is None:
			channels_to_process = [2]
		# 储存的容器
		outfile = [None, None, None]
		# outcount 用于记录每个通道录制的帧数。
		# t0 用于记录时间，用来定期报告帧数。
		# cam_names 是摄像头的名称，对应左右摄像头和拼接图像。
		outcount = [0] * len(outfile)
		t0 = time.time()
		cam_names = ['left', 'right', 'stitched']

		# loop
		while not rospy.core.is_shutdown():

			## stitching ##

			# if stitching
			if not self.image_stitcher is None:

				# stitch
				if not self.input_camera[0] is None and not self.input_camera[1] is None:
					images = [self.input_camera[0], self.input_camera[1]]
					self.input_camera[2] = cv2.hconcat(images)
					self.input_camera[0] = None
					self.input_camera[1] = None

			# for each channel to process
			for index in channels_to_process:

				# get image
				image = self.input_camera[index]

				# if present
				if not image is None:

					# handle
					self.input_camera[index] = None

					# april tags
					if not self.april_detector is None:
						image = self.detect_april(image)

					# if show
					if self.mode == "show":

						# show
						cv2.imshow("client_video: " + cam_names[index], image)
						cv2.waitKey(1)

					# if record
					if self.mode == "record":

						# create output object
						if outfile[index] is None:

							# assume fps since we don't measure it
							fps = 15

							# create
							outfile[index] = cv2.VideoWriter( \
									'/tmp/client_video_' + cam_names[index] + '.avi', \
									cv2.VideoWriter_fourcc('M','J','P','G'), \
									fps, (image.shape[1], image.shape[0]))

						# record
						outfile[index].write(image)

						# count
						outcount[index] += 1

						# report
						t1 = time.time()
						if (t1 - t0) > 1.0:
							t0 += 1.0
							print ("frames so far", outcount)

			# state
			time.sleep(0.02)

		# for each camera
		for index in range(len(outfile)):

			# if open, release
			if not outfile[index] is None:
				outfile[index].release()

	def __init__(self, args):

		# state
		self.mode = None
		self.input_camera = [None, None, None]
		self.april_detector = None
		self.image_stitcher = None

		# handle arguments
		for arg in args:
			if arg in ["show", "record"]:
				self.mode = arg
				continue
			if arg == "--april":
				self.april_detector = True
				continue
			if arg == "--stitch":
				self.image_stitcher = True
				continue
			print ("argument unrecognised:", arg)
			exit()

		# check mode
		if self.mode is None:
			error("mode not set")

		# handle april
		if not self.april_detector is None:
			if "apriltag" not in sys.modules:
				raise ValueError("April Tags library not available")
			options = apriltag.DetectorOptions( \
					families='tag16h5',
					border=1,
					nthreads=4,
					quad_decimate=1.0,
					quad_blur=0.0,
					refine_edges=True,
					refine_decode=False,
					refine_pose=False,
					debug=False,
					quad_contours=True)
			self.april_detector = apriltag.Detector(options)

		# ROS -> OpenCV converter
		self.image_converter = CvBridge()

		# robot name
		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		# subscribe
		self.sub_caml = rospy.Subscriber(topic_base_name + "/sensors/caml/compressed",
					CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
		self.sub_camr = rospy.Subscriber(topic_base_name + "/sensors/camr/compressed",
					CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)

		# report
		print ("recording from 2 cameras, press CTRL+C to halt...")

if __name__ == "__main__":

	rospy.init_node("client_video", anonymous=True)
	main = client(sys.argv[1:])
	main.loop()
