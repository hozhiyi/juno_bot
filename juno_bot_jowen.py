import rospy
from threading import Thread
from project.exercise_detection_2 import ExerciseDetector

class JunoBot:
    def __init__(self):
        rospy.init_node('juno_bot')
        self.rate = rospy.Rate(10)
        rospy.loginfo("juno_bot node started")
        
        # Create an instance of ExerciseDetector
        self.detector = ExerciseDetector()

    def start(self):
        # Start the thread for the app
        app_thread = Thread(target=self.run_app)
        app_thread.start()

        # Keep the node running
        while not rospy.is_shutdown():
            self.rate.sleep()

    def run_app(self):
        # Run the exercise detection
        self.detector.start()

if __name__ == "__main__":
    bot = JunoBot()
    bot.start()
