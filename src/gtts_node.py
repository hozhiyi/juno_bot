#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from std_msgs.msg import String
from gtts import gTTS
import os

class CountSubscriber:
    def __init__(self):
        rospy.init_node('tts_node', anonymous=True)

        # Create subscribers for the count topics
        rospy.Subscriber('/exercise_counters/curl', Int32, self.curl_callback)
        rospy.Subscriber('/exercise_counters/press', Int32, self.press_callback)
        rospy.Subscriber('/exercise_counters/squat', Int32, self.squat_callback)


    def count_to_speech(self, count):
        rospy.loginfo("Input: %s", count)
        # Convert the count to speech using gtts
        tts = gTTS(text=str(count), lang='en')
        
        tts.save("speech.mp3")
        os.system("mpg321 speech.mp3")
        os.remove("speech.mp3")
       
    def curl_callback(self, msg):
        count = msg.data
        text = "curl {}".format(count)
        self.count_to_speech(text)

    def press_callback(self, msg):
        count = msg.data
        text = "press {}".format(count)
        self.count_to_speech(text)

    def squat_callback(self, msg):
        count = msg.data
        text = "squat {}".format(count)
        self.count_to_speech(text)


if __name__ == '__main__':
    try:
        CountSubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
