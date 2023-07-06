#!/usr/bin/env python

import numpy as np
import os
import tensorflow as tf
import cv2
import mediapipe as mp
import math
from threading import Thread



import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2
from std_msgs.msg import Int32

class ExerciseDetector2:
    def __init__(self,model_path):
        # Initialize the ROS node
        print('** initialized')

        rospy.init_node('exercise_detection')
        rospy.loginfo('hi')

        print('** initialized rospy')

        # Create publishers for the counter values
        self.curl_counter_pub = rospy.Publisher('/exercise_counters/curl', Int32, queue_size=10)
        self.press_counter_pub = rospy.Publisher('/exercise_counters/press', Int32, queue_size=10)
        self.squat_counter_pub = rospy.Publisher('/exercise_counters/squat', Int32, queue_size=10)
        
        self.model = tf.keras.models.load_model(model_path)
        print('** loaded model')

        # Pre-trained pose estimation model from Google Mediapipe
        self.mp_pose = mp.solutions.pose
        # Supported Mediapipe visualization tools
        self.mp_drawing = mp.solutions.drawing_utils
            # Colors associated with each exercise (e.g., curls are denoted by blue, squats are denoted by orange, etc.)
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        
        self.bridge = CvBridge()
        
        # Create an OpenCV window
        cv2.namedWindow("Exercise Detection")

        print('** completed init function')

    # 1. Keypoints detection
    def mediapipe_detection(self,image, model):
        """
        This function detects human pose estimation keypoints from webcam footage
        
        """
        print('** starts mediapipe_detection')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        print('** done mediapipe_detection')
        return image, results
    
    def draw_landmarks(self,image, results):
        """
        This function draws keypoints and landmarks detected by the human pose estimation model
        
        """
        print('** start draw_landmarks')
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
        print('** done draw_landmarks')

    # 2. Keypoints extraction
    def extract_keypoints(self,results):
        """
        Processes and organizes the keypoints detected from the pose estimation model 
        to be used as inputs for the exercise decoder models
        
        """
        print('** start extract_keypoints')
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        print('** done extract_keypoints')
        return pose
    
    # 3. Calculate Joint Angles and Count Reps
    def get_coordinates(self,landmarks, mp_pose, side, joint):
        """
        Retrieves x and y coordinates of a particular keypoint from the pose estimation model
            
        Args:
            landmarks: processed keypoints from the pose estimation model
            mp_pose: Mediapipe pose estimation model
            side: 'left' or 'right'. Denotes the side of the body of the landmark of interest.
            joint: 'shoulder', 'elbow', 'wrist', 'hip', 'knee', or 'ankle'. Denotes which body joint is associated with the landmark of interest.
        
        """
        print('** start get_coordinates')
        coord = getattr(mp_pose.PoseLandmark,side.upper()+"_"+joint.upper())
        x_coord_val = landmarks[coord.value].x
        y_coord_val = landmarks[coord.value].y
        print('** done get_coordinates')
        return [x_coord_val, y_coord_val]    
    
    def viz_joint_angle(self,image, angle, joint):
        """
        Displays the joint angle value near the joint within the image frame
        
        """
        print('** start viz_joint_angle')
        cv2.putText(image, str(int(angle)), 
                    tuple(np.multiply(joint, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
        print('** done viz_joint_angle')
        
        return
    
    def calculate_angle(self,a,b,c):
        """
        Computes 3D joint angle inferred by 3 keypoints and their relative positions to one another
        
        """
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle 

    def count_reps(self,image, current_action, landmarks, mp_pose):
        """
        Counts repetitions of each exercise. Global count and stage (i.e., state) variables are updated within this function.
        
        """

        global curl_counter, press_counter, squat_counter, curl_stage, press_stage, squat_stage
        
        if current_action == 'curl':
            # Get coords
            shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
            wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')
            
            # calculate elbow angle
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # curl counter logic
            if angle < 30:
                curl_stage = "up" 
            if angle > 140 and curl_stage =='up':
                curl_stage="down"  
                curl_counter +=1
                self.curl_counter_pub.publish(curl_counter)
            press_stage = None
            squat_stage = None
                
            # Viz joint angle
            self.viz_joint_angle(image, angle, elbow)
            
        elif current_action == 'press':
            
            # Get coords
            shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
            wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')

            # Calculate elbow angle
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # Compute distances between joints
            shoulder2elbow_dist = abs(math.dist(shoulder,elbow))
            shoulder2wrist_dist = abs(math.dist(shoulder,wrist))
            
            # Press counter logic
            if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist):
                press_stage = "up"
            if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (press_stage =='up'):
                press_stage='down'
                press_counter += 1
                self.press_counter_pub.publish(press_counter)
            curl_stage = None
            squat_stage = None
                
            # Viz joint angle
            self.viz_joint_angle(image, elbow_angle, elbow)
            
        elif current_action == 'squat':
            # Get coords
            # left side
            left_shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            left_hip = self.get_coordinates(landmarks, mp_pose, 'left', 'hip')
            left_knee = self.get_coordinates(landmarks, mp_pose, 'left', 'knee')
            left_ankle = self.get_coordinates(landmarks, mp_pose, 'left', 'ankle')
            # right side
            right_shoulder = self.get_coordinates(landmarks, mp_pose, 'right', 'shoulder')
            right_hip = self.get_coordinates(landmarks, mp_pose, 'right', 'hip')
            right_knee = self.get_coordinates(landmarks, mp_pose, 'right', 'knee')
            right_ankle = self.get_coordinates(landmarks, mp_pose, 'right', 'ankle')
            
            # Calculate knee angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Calculate hip angles
            left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Squat counter logic
            thr = 165
            if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (right_hip_angle < thr):
                squat_stage = "down"
            if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (right_hip_angle > thr) and (squat_stage =='down'):
                squat_stage='up'
                squat_counter += 1
                self.squat_counter_pub.publish(squat_counter) 
            curl_stage = None
            press_stage = None
                
            # Viz joint angles
            self.viz_joint_angle(image, left_knee_angle, left_knee)
            self.viz_joint_angle(image, left_hip_angle, left_hip)
            
        else:
            pass

    # 4. Run the exercise detection app
    def prob_viz(self, res, actions, input_frame, colors):
        """
        This function displays the model prediction probability distribution over the set of exercise classes
        as a horizontal bar graph
        
        """
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):        
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame
    
    def image_callback(self,msg):
        
        print('** start image_callback')

        sequence = []
        predictions = []
        res = []
        threshold = 0.5 # minimum confidence to classify as an action/exercise
        current_action = ''

        # Rep counter logic variables
        global curl_counter, press_counter, squat_counter, curl_stage, press_stage, squat_stage
        # Rep counter logic variables
        curl_counter = 0
        press_counter = 0
        squat_counter = 0
        curl_stage = None
        press_stage = None
        squat_stage = None
        try:
            cap = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Process the image (mediapipe detection, keypoints extraction, counting reps, etc.)
        except CvBridgeError as e:
            print(e)
        
        # print(cap)
        # while(True):
        # # Capture the video frame
        # # by frame
        #     # ret, frame = cap.read()
        #     # Display the resulting frame
        #     cv2.imshow('frame', cap)
        #     # the 'q' button is set as the
        #     # quitting button you may use any
        #     # desired button of your choice
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # # Destroy all the windows
        # cv2.destroyAllWindows()

        # Video writer object that saves a video of the real time test
        #testing
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') # video compression format
        # HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # webcam video frame height
        # WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # webcam video frame width
        # FPS = int(cap.get(cv2.CAP_PROP_FPS)) # webcam video fram rate 
        #temporary set value cuz cant use cap.get
        HEIGHT = 100
        WIDTH = 100
        FPS = 30
        # Videos are going to be this many frames in length
        sequence_length = FPS*1

        # Actions/exercises that we try to detect
        actions = np.array(['curl', 'press', 'squat'])

        # video_name = os.path.join(os.getcwd(),f"{model_name}_real_time_test.avi")
        video_name = os.path.join(os.getcwd(),"LSTM_Attention_real_time_test.avi")
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH,HEIGHT))


        # Set mediapipe model 
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # while cap.isOpened():
            while cap.all()!=0:

                # Read feed
                # ret, frame = cap.read()
                frame = cap

                # Make detection
                image, results = self.mediapipe_detection(frame, pose)
                
                # Draw landmarks
                self.draw_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = self.extract_keypoints(results)        
                sequence.append(keypoints)      
                sequence = sequence[-sequence_length:]
                    
                if len(sequence) == sequence_length:
                    res = self.model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]           
                    predictions.append(np.argmax(res))
                    current_action = actions[np.argmax(res)]
                    confidence = np.max(res)
                    
                #3. Viz logic
                    # Erase current action variable if no probability is above threshold
                    if confidence < threshold:
                        current_action = ''

                    # Viz probabilities
                    image = self.prob_viz(res, actions, image, self.colors)
                    
                    # Count reps
                    try:
                        landmarks = results.pose_landmarks.landmark
                        self.count_reps(
                            image, current_action, landmarks, self.mp_pose)
                    except:
                        pass

                    # Display graphical information
                    cv2.rectangle(image, (0,0), (640, 40), self.colors[np.argmax(res)], -1)
                    cv2.putText(image, 'curl ' + str(curl_counter), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'press ' + str(press_counter), (240,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'squat ' + str(squat_counter), (490,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                # Write to video file

                cv2.imshow('Exercise Detection', image)

                # if ret == True:
                if cap.all()!= 0:
                    out.write(image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            # cap.release()
            out.release()
            cv2.destroyAllWindows()
    

    def run_app(self):
        
        print('** start run_app')
        rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        
        print('** done run_app')

        
    def start(self):
        
        print('** start start')
        # Start the exercise detection app in a separate thread
        thread = Thread(target=self.run_app)
        thread.start()
        print('** done start')



if __name__ == "__main__":
    try:
        print('** checkpoint 1')
        # Provide the path to the saved TensorFlow model
        model_path = 'models/LSTM_Attention.h5'

        print('** checkpoint 2')
        # Create an instance of ExerciseDetector and start the app
        detector = ExerciseDetector2(model_path)

        print('** checkpoint 3')
        detector.start()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Bot Terminated")
