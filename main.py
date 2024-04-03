import cv2

captureSource = 0
capture = cv2.VideoCapture(captureSource)

flipFlag = True

while True:
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break
    
    has_frame, frame = capture.read()

    if key == ord('f') or key == ord('F'):
        flipFlag = not flipFlag

    if flipFlag:
        frame = cv2.flip(frame, 1)

    if not has_frame:
        break
    
    cv2.imshow('Video Camera', frame)
    
capture.release()
cv2.destroyAllWindows()
