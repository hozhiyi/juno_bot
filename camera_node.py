#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
       

if __name__ == '__main__':
    
    rospy.init_node('camera_node', anonymous=True)
    pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
    # 0 is laptop camera
    # 2 is juno camera
    camera_resource = 2
    print(f"Trying to open resource: {camera_resource}")
    cap = cv2.VideoCapture(camera_resource)
    bridge = CvBridge()
    rate = rospy.Rate(2)  # Set the desired frame rate

    if not cap.isOpened():
        print("Error opening resource: " + str(camera_resource))
        print("Maybe opencv VideoCapture can't open it")
        exit(0)
    
    print("Correctly opened resource, starting to show feed.")
    try:
        while not rospy.is_shutdown():
            ret, frame = cap.read()

            if ret:
                image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
                pub.publish(image_msg)

            rate.sleep()
    finally:
        if cap is not None:
            cap.release()

    rospy.loginfo("Node was stopped")
