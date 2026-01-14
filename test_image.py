from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model("image/test.jpg",show = True)