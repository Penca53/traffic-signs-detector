from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("stop-sign.pt")

    model.predict("datasets/stop_signs/images/val/81d07c50-downloadstop.jpg", save=True)