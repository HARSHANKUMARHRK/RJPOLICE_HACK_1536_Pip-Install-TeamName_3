import cv2

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
c = 0

displacement_threshold = 20

while True:
    ret, next_frame = cap.read()

    if not ret:
        break

    frame_diff = cv2.absdiff(prev_frame, next_frame)
    frame_diff_gray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2RGB)

    mean_diff = cv2.mean(frame_diff_gray)[0]

    if mean_diff > displacement_threshold:
        c += 1
        print(c)
        print("Camera displaced")
        cv2.imwrite(f'displacement_{c}.jpg', prev_frame)
        
    else:
        cv2.imshow('Original Feed', next_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_frame = next_frame.copy()

cap.release()
