from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("extended-signs.pt")

    model.predict("https://www.ilankelman.org/stopsigns/latvia.jpg", save=True)