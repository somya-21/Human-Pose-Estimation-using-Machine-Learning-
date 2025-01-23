import cv2
import mediapipe as mp
import numpy as np
import time

class PoseEstimator:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_pose(self, img, draw=True):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks and draw:
            # Draw landmarks and connections
            self.mp_draw.draw_landmarks(
                img, 
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        return img
    
    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # Convert landmark coordinates to pixel values
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lm_list

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize pose detector
    detector = PoseEstimator()
    
    # Initialize FPS calculation variables
    p_time = 0
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        # Find pose
        img = detector.find_pose(img)
        
        # Find position
        lm_list = detector.find_position(img)
        
        # Calculate and display FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        
        # Display the image
        cv2.imshow("Pose Estimation", img)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()