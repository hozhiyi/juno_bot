# create a ros architecture that runs pose estimation project 
import rospy
import cv2
import numpy as np
import tensorflow as tf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import mediapipe as mp
import time
import camera

if __name__ == '__main__':
    rospy.init_node('juno_bot')
    rate = rospy.Rate(10)
    rospy.loginfo("juno_bot node started")

    main()
    
    rospy.loginfo("juno_bot node terminated")


def main():

    # 1. New detection variables
    sequence = []
    predictions = []
    res = []
    threshold = 0.5 # minimum confidence to classify as an action/exercise
    current_action = ''

    # Rep counter logic variables
    curl_counter = 0
    press_counter = 0
    squat_counter = 0
    curl_stage = None
    press_stage = None
    squat_stage = None

    # Camera object
    # cap = cv2.VideoCapture(0)
    # define Image for rospy subscriber
    image = Image()
    # use the rospy camera subscriber from camara.py instead of cv2.VideoCapture 
    camera_subscriber = CameraSubscriber()
    camera_subscriber.run() 
    


    # Video writer object that saves a video of the real time test
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G') # video compression format
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # webcam video frame height
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # webcam video frame width
    FPS = int(cap.get(cv2.CAP_PROP_FPS)) # webcam video fram rate 

    video_name = os.path.join(os.getcwd(),f"{model_name}_real_time_test.avi")
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH,HEIGHT))

    # Set mediapipe model 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detection
            image, results = mediapipe_detection(frame, pose)
            
            # Draw landmarks
            draw_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)        
            sequence.append(keypoints)      
            sequence = sequence[-sequence_length:]
                
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]           
                predictions.append(np.argmax(res))
                current_action = actions[np.argmax(res)]
                confidence = np.max(res)
                
            #3. Viz logic
                # Erase current action variable if no probability is above threshold
                if confidence < threshold:
                    current_action = ''

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)
                
                # Count reps
                try:
                    landmarks = results.pose_landmarks.landmark
                    count_reps(
                        image, current_action, landmarks, mp_pose)
                except:
                    pass

                # Display graphical information
                cv2.rectangle(image, (0,0), (640, 40), colors[np.argmax(res)], -1)
                cv2.putText(image, 'curl ' + str(curl_counter), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'press ' + str(press_counter), (240,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'squat ' + str(squat_counter), (490,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            
            # Write to video file
            if ret == True:
                out.write(image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
