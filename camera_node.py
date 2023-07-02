#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode:
    def __init__(self):
        rospy.init_node('camera_node', anonymous=True)
        self.pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
        self.cap = cv2.VideoCapture(4)
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30)  # Set the desired frame rate

    def capture_frames(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()

            if ret:
                image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.pub.publish(image_msg)

            self.rate.sleep()

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()

if __name__ == '__main__':
    camera_node = CameraNode()
    try:
        camera_node.capture_frames()
    finally:
        camera_node.release_camera()
