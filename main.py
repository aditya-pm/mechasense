from ultralytics import YOLO
import cv2
import webbrowser

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# prevents multiple browser openings
browser_opened = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    # get detected class IDs
    classes = results[0].boxes.cls.tolist() if results[0].boxes else []

    # convert IDs to class names
    names = [model.names[int(c)] for c in classes]

    # check if laptop detected
    if "laptop" in names:
        cv2.putText(
            annotated,
            "Laptop detected - Opening 3D disassembly",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # open viewer only once
        if not browser_opened:
            webbrowser.open(
                "file:///Users/aditya/ProgrammingProjects/PythonProjects/ObjectDetection/viewer.html"
            )  # path to your A-Frame file
            browser_opened = True

    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
