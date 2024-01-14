import cv2
from roboflow import Roboflow
rf = Roboflow(api_key="1XpjtVozI07Y2KQN12Mw")
project = rf.workspace().project("natural-disaster-damage-8txjn")
model = project.version(1).model
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = model.predict(rgb_frame).json()

    if "predictions" in predictions:

        for prediction in predictions["predictions"]:
            label = prediction["label"]
            confidence = prediction["confidence"]
            bbox = prediction["bbox"]
            x, y, w, h = bbox["xmin"], bbox["ymin"], bbox["width"], bbox["height"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label_text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Video Feed with Predictions", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
