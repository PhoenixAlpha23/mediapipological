import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

class FootballPoseAnalyzer:
    def __init__(self):
        # Initialize YOLO for player detection
        self.player_detector = YOLO('yolov8n.pt')
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def process_frame(self, frame):
        # 1. Detect players using YOLO
        player_results = self.player_detector(frame, classes=[0])  # class 0 is person
        
        # 2. Process each detected player with MediaPipe
        annotated_frame = frame.copy()
        
        for box in player_results[0].boxes.xyxy:
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            
            # Extract player ROI
            player_roi = frame[y1:y2, x1:x2]
            if player_roi.size == 0:
                continue
                
            # Process pose
            results = self.pose.process(cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # Draw pose landmarks on ROI
                self.mp_draw.draw_landmarks(
                    player_roi,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
            # Put processed ROI back into frame
            annotated_frame[y1:y2, x1:x2] = player_roi
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        return annotated_frame
    
    def __del__(self):
        self.pose.close()

# Usage example
def main():
    cap = cv2.VideoCapture(0)  # Replace with video file path for football footage
    analyzer = FootballPoseAnalyzer()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        annotated_frame = analyzer.process_frame(frame)
        
        cv2.imshow('Football Pose Analysis', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()