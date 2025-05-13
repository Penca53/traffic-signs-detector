from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    results = model.train(data="datasets/stop_signs/stop_signs.yaml", epochs=100, device=0)