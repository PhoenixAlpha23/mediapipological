from ultralytics import YOLO

class PlayerDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # or custom trained model
        
    def detect_players(self, frame):
        results = self.model(frame, classes=[0])  # class 0 is person
        return results[0].boxes.xyxy.cpu().numpy()