from ultralytics import YOLO, settings

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    results = model.train(data="datasets/V2_Traffic_Signs/data.yaml", epochs=100, device=0)