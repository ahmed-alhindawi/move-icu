import mediapipe as mp
import cv2
import numpy as np

def standalone_pose_detection():
    '''
    Function that uses MediaPipe Pose Landmarker to estimate the position of key body locations 
    on a livestream of input images from the webcam

    For MediaPipe Pose Landmarker see https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
    '''

    # Webcam video feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Import mediapipe model and drawing utilities
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            # Capture video frame
            ret, frame = cap.read()
            if not ret:
                    print("Error: Failed to capture frame.")
                    break
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Pose detection
            results = pose.process(image)

            # Color back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                 landmarks = results.pose_landmarks.landmark
                 #print(landmarks)
            except:
                 pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display webcam feed with detected landmarks/skeleton - press 'Q' to exit
            cv2.imshow("Mediapipe Feed", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
     standalone_pose_detection()