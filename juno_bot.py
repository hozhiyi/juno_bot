import rospy
import cv2
import numpy as np
import tensorflow as tf

class juno_bot:
	def __init__(self):
        
        rospy.init_node('juno_bot')
        self.rate = rospy.Rate(10)
        rospy.loginfo("juno_bot node started")
		
    
    def load_model(self, model_path):
        
        model = tf.keras.models.load_model(model_path) 
	
    
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


    def get_coordinates(self, landmarks, mp_pose, side, joint):
        """
        Retrieves x and y coordinates of a particular keypoint from the pose estimation model
            
        Args:
            landmarks: processed keypoints from the pose estimation model
            mp_pose: Mediapipe pose estimation model
            side: 'left' or 'right'. Denotes the side of the body of the landmark of interest.
            joint: 'shoulder', 'elbow', 'wrist', 'hip', 'knee', or 'ankle'. Denotes which body joint is associated with the landmark of interest.
        
        """
        coord = getattr(mp_pose.PoseLandmark,side.upper()+"_"+joint.upper())
        x_coord_val = landmarks[coord.value].x
        y_coord_val = landmarks[coord.value].y
        return [x_coord_val, y_coord_val] 
    

    def viz_joint_angle(image, angle, joint):
        """
        Displays the joint angle value near the joint within the image frame
        
        """
        cv2.putText(image, str(int(angle)), 
                    tuple(np.multiply(joint, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
        return
    

    def count_reps(image, current_action, landmarks, mp_pose):
        """
        Counts repetitions of each exercise. Global count and stage (i.e., state) variables are updated within this function.
        
        """

        global curl_counter, press_counter, squat_counter, curl_stage, press_stage, squat_stage
        
        if current_action == 'curl':
            # Get coords
            shoulder = get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            elbow = get_coordinates(landmarks, mp_pose, 'left', 'elbow')
            wrist = get_coordinates(landmarks, mp_pose, 'left', 'wrist')
            
            # calculate elbow angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # curl counter logic
            if angle < 30:
                curl_stage = "up" 
            if angle > 140 and curl_stage =='up':
                curl_stage="down"  
                curl_counter +=1
            press_stage = None
            squat_stage = None
                
            # Viz joint angle
            viz_joint_angle(image, angle, elbow)
            
        elif current_action == 'press':
            
            # Get coords
            shoulder = get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            elbow = get_coordinates(landmarks, mp_pose, 'left', 'elbow')
            wrist = get_coordinates(landmarks, mp_pose, 'left', 'wrist')

            # Calculate elbow angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            
            # Compute distances between joints
            shoulder2elbow_dist = abs(math.dist(shoulder,elbow))
            shoulder2wrist_dist = abs(math.dist(shoulder,wrist))
            
            # Press counter logic
            if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist):
                press_stage = "up"
            if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (press_stage =='up'):
                press_stage='down'
                press_counter += 1
            curl_stage = None
            squat_stage = None
                
            # Viz joint angle
            viz_joint_angle(image, elbow_angle, elbow)
            
        elif current_action == 'squat':
            # Get coords
            # left side
            left_shoulder = get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            left_hip = get_coordinates(landmarks, mp_pose, 'left', 'hip')
            left_knee = get_coordinates(landmarks, mp_pose, 'left', 'knee')
            left_ankle = get_coordinates(landmarks, mp_pose, 'left', 'ankle')
            # right side
            right_shoulder = get_coordinates(landmarks, mp_pose, 'right', 'shoulder')
            right_hip = get_coordinates(landmarks, mp_pose, 'right', 'hip')
            right_knee = get_coordinates(landmarks, mp_pose, 'right', 'knee')
            right_ankle = get_coordinates(landmarks, mp_pose, 'right', 'ankle')
            
            # Calculate knee angles
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Calculate hip angles
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Squat counter logic
            thr = 165
            if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (right_hip_angle < thr):
                squat_stage = "down"
            if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (right_hip_angle > thr) and (squat_stage =='down'):
                squat_stage='up'
                squat_counter += 1
            curl_stage = None
            press_stage = None
                
            # Viz joint angles
            viz_joint_angle(image, left_knee_angle, left_knee)
            viz_joint_angle(image, left_hip_angle, left_hip)
            
        else:
            pass
    

    def prob_viz(res, actions, input_frame, colors):
        """
        This function displays the model prediction probability distribution over the set of exercise classes
        as a horizontal bar graph
        
        """
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):        
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame
    