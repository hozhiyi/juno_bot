#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
       

if __name__ == '__main__':
    
    rospy.init_node('camera_node', anonymous=True)
    pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
    # 0 is laptop camera
    # 2 is juno camera
    camera_resource = 0
    print("Trying to open resource: ")
    # print(f"Trying to open resource: {camera_resource}")
    cap = cv2.VideoCapture(camera_resource)

    # print(cap)
    # while(True):
    # # Capture the video frame
    # # by frame
    #     ret, frame = cap.read()
    #     # Display the resulting frame
    #     cv2.imshow('frame', frame)
    #     # the 'q' button is set as the
    #     # quitting button you may use any
    #     # desired button of your choice
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # Destroy all the windows
    # cv2.destroyAllWindows()
        
    bridge = CvBridge()
    rate = rospy.Rate(10)  # Set the desired frame rate

    if not cap.isOpened():
        print("Error opening resource: " )
        print("Error opening resource: " + str(camera_resource))
        print("Maybe opencv VideoCapture can't open it")
        exit(0)
    
    print("Correctly opened resource, starting to show feed.")
    try:
        while not rospy.is_shutdown():
            ret, frame = cap.read()

            if ret:
                try:
                    image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
                except CvBridgeError as e:
                    print(e)
                pub.publish(image_msg)

            rate.sleep()
    finally:
        if cap is not None:
            cap.release()

    rospy.loginfo("Node was stopped")