import os
import cv2
from ultralytics import YOLOWorld

def main():
    print("Loading YOLO-World model (yolov8x-worldv2.pt)...")
    model = YOLOWorld("/workspace/yolov8x-worldv2.pt")
    
    # Set classes
    classes = ["stairs", "staircase", "steps", "brick stairs", "brick steps", "brick", "brick pattern", "person"]
    model.set_classes(classes)
    print(f"Classes set to: {classes}")
    
    # Path to the textured stairs image
    img_path = "/workspace/log/verification_test.png"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} does not exist.")
        return
        
    img = cv2.imread(img_path)
    print("Running prediction...")
    results = model.predict(img, conf=0.001, verbose=False)
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            print(f"Success! Detected {len(boxes)} object(s):")
            for idx, box in enumerate(boxes):
                cls_idx = int(box.cls[0].cpu().item())
                class_name = results[0].names[cls_idx]
                conf = float(box.conf[0].cpu().item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                print(f"  [{idx}] Class: '{class_name}', Conf: {conf:.4f}, BBox: {xyxy}")
        else:
            print("No objects detected (0 boxes).")
    else:
        print("No prediction results returned.")

if __name__ == "__main__":
    main()
