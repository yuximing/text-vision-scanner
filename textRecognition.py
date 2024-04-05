import cv2
import cv2.rgbd
import matplotlib.pyplot as plt
from textDetection import getDetectionPoints
import cv2.cuda

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

def recognizeText():
    # load image
    image = cv2.imread("images/fortunecow.png", cv2.IMREAD_COLOR)
    
    net = cv2.dnn.readNet('crnn_cs_CN.onnx')
    model = cv2.dnn.TextRecognitionModel(net)
    
    # decode 
    model.setDecodeType("CTC-greedy")

    with open("alphabet_3944.txt", "r", encoding="utf8") as file:
        vocabulary = [line.rstrip('\n') for line in file.readlines()]
        model.setVocabulary(vocabulary)
    
    # Normalization parameters
    scale = 1.0 / 127.5
    mean = (127.5, 127.5, 127.5)
    
    # The input shape
    inputSize = (100, 32)
    
    model.setInputParams(scale, inputSize, mean)

    # std::string recognitionResult = recognizer.recognize(image);
    # std::cout << "'" << recognitionResult << "'" << std::endl;
    # load model
    net_detect = cv2.dnn.readNet('frozen_east_text_detection.pb')
    model_detect = cv2.dnn.TextDetectionModel_EAST(net_detect)
    (detections, confidence) = getDetectionPoints(model_detect, image)

    out = model.recognize(image, detections)
    print(out)
    
if __name__ == "__main__":
    # main()
    recognizeText()

