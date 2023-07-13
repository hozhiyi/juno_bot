#!/usr/bin/env python

import numpy as np
import os
import tensorflow as tf
import cv2
import mediapipe as mp
import math


import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge,CvBridgeError


class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber')
        rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("Camera", image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)

class ExerciseDetector2(ImageSubscriber):
    
    def __init__(self, model_path):
        # Initialize counters for each exercise
         # Create publishers for the counter values
        self.curl_counter_pub = rospy.Publisher('/exercise_counters/curl', Int32, queue_size=10)
        self.press_counter_pub = rospy.Publisher('/exercise_counters/press', Int32, queue_size=10)
        self.squat_counter_pub = rospy.Publisher('/exercise_counters/squat', Int32, queue_size=10)
        self.model = tf.keras.models.load_model(model_path)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
        # self.bridge = CvBridge()

        self.curl_counter = 0
        self.press_counter = 0
        self.squat_counter = 0
        self.curl_stage = None
        self.press_stage = None
        self.squat_stage = None
        self.sequence_length = 30  # Define the desired sequence length
        self.actions = np.array(['curl', 'press', 'squat'])  # Define the actions/exercises
        
        self.sequence = []
        self.predictions = []
        self.res = []
        self.threshold = 0.5
        self.current_action = ''

        super().__init__()  # Initialize the superclass




    def mediapipe_detection(self, image, model):
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

    def draw_landmarks(self, image, results):
        """
        This function draws keypoints and landmarks detected by the human pose estimation model
        
        """
        print('** start draw_landmarks')
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
        print('** done draw_landmarks')

    def extract_keypoints(self, results):
        """
        Processes and organizes the keypoints detected from the pose estimation model 
        to be used as inputs for the exercise decoder models
        
        """
        print('** start extract_keypoints')
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        print('** done extract_keypoints')
        return pose

    def get_coordinates(self, landmarks, mp_pose, side, joint):
        """
        Retrieves x and y coordinates of a particular keypoint from the pose estimation model
            
        Args:
            landmarks: processed keypoints from the pose estimation model
            mp_pose: Mediapipe pose estimation model
            side: 'left' or 'right'. Denotes the side of the body of the landmark of interest.
            joint: 'shoulder', 'elbow', 'wrist', 'hip', 'knee', or 'ankle'. Denotes which body joint is associated with the landmark of interest.
        
        """
        print('** start get_coordinates')
        coord = getattr(self.mp_pose.PoseLandmark,side.upper()+"_"+joint.upper())
        x_coord_val = landmarks[coord.value].x
        y_coord_val = landmarks[coord.value].y
        print('** done get_coordinates')
        return [x_coord_val, y_coord_val]    

    def viz_joint_angle(self, image, angle, joint):
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

    def calculate_angle(self, a, b, c):
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


    def count_reps(self, image, current_action, landmarks, mp_pose):
        # """
        # Counts repetitions of each exercise. Global count and stage (i.e., state) variables are updated within this function.
        # """
        if current_action == 'curl':
                # Get coords
                shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
                elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
                wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')
                
                # calculate elbow angle
                angle = self.calculate_angle(shoulder, elbow, wrist)
                # curl counter logic
                
                if angle < 30:
                    self.curl_stage = "up"
                
                if angle > 70 and self.curl_stage =='up':
                    self.curl_stage="down"  
                    self.curl_counter +=1
                    self.curl_counter_pub.publish(self.curl_counter)

                self.press_stage = None
                self.squat_stage = None 
                                
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
                self.press_stage = "up"
            if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (self.press_stage =='up'):
                self.press_stage='down'
                self.press_counter += 1
                self.press_counter_pub.publish(self.press_counter)
            self.curl_stage = None
            self.squat_stage = None
                
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
                self.squat_stage = "down"
            if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (right_hip_angle > thr) and (self.squat_stage =='down'):
                self.squat_stage='up'
                self.squat_counter += 1
                self.squat_counter_pub.publish(self.squat_counter) 
            self.curl_stage = None
            self.press_stage = None
                
            # Viz joint angles
            self.viz_joint_angle(image, left_knee_angle, left_knee)
            self.viz_joint_angle(image, left_hip_angle, left_hip)
            
        else:
            pass


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

    def image_callback(self, msg):


        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        #Make detection
        image, results = self.mediapipe_detection(frame, pose)

        #Draw landmarks
        self.draw_landmarks(image, results)

        #Prediction logic
        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-self.sequence_length:]

        # confidence = 0.0  # Initialize confidence with a default value
        if len(self.sequence) == self.sequence_length:
            self.res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            self.predictions.append(np.argmax(self.res))
            current_action = self.actions[np.argmax(self.res)]
            confidence = np.max(self.res)

            if confidence < self.threshold:
                current_action = ''

            image = self.prob_viz(self.res, self.actions, image, self.colors)

            try:
                landmarks = results.pose_landmarks.landmark
                self.count_reps(image, current_action, landmarks, self.mp_pose)
            except:
                pass
        
 
        # if len(self.res)!=0:    
        # Display graphical information
            cv2.rectangle(image, (0,0), (640, 40), self.colors[np.argmax(self.res)], -1)
            cv2.putText(image, 'curl ' + str(self.curl_counter), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'press ' + str(self.press_counter), (240,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'squat ' + str(self.squat_counter), (490,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Exercise Detection', image)
        cv2.waitKey(1)




if __name__ == "__main__":
    try:
        model_path = 'models/LSTM_Attention.h5'
        detector = ExerciseDetector2(model_path)
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Bot Terminated")
