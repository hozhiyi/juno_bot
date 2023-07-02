import rospy
from std_msgs.msg import Int32
import pyttsx3

class TextToSpeech:
    def __init__(self):
        # Initialize the text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Adjust the speech rate if needed
        self.engine.setProperty('volume', 1.0)  # Adjust the volume if needed

        # Subscribe to the exercise counter topics
        rospy.Subscriber('/exercise_counters/curl', Int32, self.curl_counter_callback)
        rospy.Subscriber('/exercise_counters/press', Int32, self.press_counter_callback)
        rospy.Subscriber('/exercise_counters/squat', Int32, self.squat_counter_callback)

    def text_to_speech(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def curl_counter_callback(self, msg):
        count = msg.data
        text = "curl {}".format(count)
        self.text_to_speech(text)

    def press_counter_callback(self, msg):
        count = msg.data
        text = "press {}".format(count)
        self.text_to_speech(text)

    def squat_counter_callback(self, msg):
        count = msg.data
        text = "squat {}".format(count)
        self.text_to_speech(text)

if __name__ == "__main__":
    rospy.init_node('text_to_speech_node', anonymous=True)

    tts = TextToSpeech()

    rospy.spin()
