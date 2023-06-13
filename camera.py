import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraSubscriber:
    def __init__(self):
        rospy.init_node('camera_subscriber')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('camera_topic', Image, self.image_callback)
    
    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imshow("Robot Camera", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("Error processing image: {}".format(e))
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    camera_subscriber = CameraSubscriber()
    camera_subscriber.run()
