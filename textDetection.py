import cv2
import cv2.rgbd
import matplotlib.pyplot as plt

def main():

    # load model
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
    model = cv2.dnn.TextDetectionModel_EAST(net)
    image = cv2.imread('images/random.png')

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
        
        (detections, confidence) = getDetectionPoints(model, frame)
        frameWithDetections = cv2.polylines(frame, detections, isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow('Video Camera', frameWithDetections)
        
    capture.release()
    cv2.destroyAllWindows()

def getDetectionPoints(model, image):
    # set parameters
    confThreshold = 0.5
    nmsThreshold = 0.4
    model.setConfidenceThreshold(confThreshold)
    model.setNMSThreshold(nmsThreshold)
    
    detScale = 1.0
    detInputSize = (320, 320)
    detMean = (123.68, 116.78, 103.94)
    swapRB = True
    model.setInputParams(detScale, detInputSize, detMean, swapRB)
    # plt.figure(figsize=[5, 5])
    # plt.imshow(image)
    # plt.show()
    return model.detect(image)

    
if __name__ == "__main__":
    main()

