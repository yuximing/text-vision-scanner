import cv2
import cv2.rgbd
import matplotlib.pyplot as plt
from textDetection import getDetectionPoints
import cv2.cuda
import numpy as np

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
    image = cv2.imread("images/eclipse.png", cv2.IMREAD_COLOR)
    
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


    words = []
    for detection in detections:
        bl = detection[0]
        tl = detection[1]
        tr = detection[2]
        br = detection[3]
        
        minh = min(bl[0], tl[0])
        maxh = max(br[0], tr[0])
        minv = min(tl[1], tr[1])
        maxv = max(bl[1], br[1])
        
        img_cropped = image[minv:maxv, minh:maxh]
        # plt.figure()
        # plt.imshow(img_cropped)
        # plt.show()
        word = model.recognize(img_cropped)
        words.append(word)

        # print(detection)
    
    orderWords(detections, image.shape) # TEST


def orderWords(detections, imageDimensions):
    # display the contours on a black bg

    # Create black background
    black_img = np.zeros((imageDimensions[0], imageDimensions[1], 1), dtype=np.int32)
    black_img = cv2.polylines(black_img, detections, isClosed=True, color=(255,), thickness=1)

    # DO NOT DELETE ME
    # [ 
    #     [ [0] [0] [0] [0]], black_img[0] =>  [ [0] [255] [0] [255]]  [  x  x]    say 3-d, [[255, 255, 255] [255, 255, 255] [0, 255, 255]] = [[255 *2+0, 255 *3, 255 *3]]
    #     [ [0] [0] [0] [0]],                  [ [255] [255] [255] [255]]  [  x  x]
    #     [ [0] [0] [0] [0]],
    #     [ [] [] [] []],
    # ]

    sum_of_rows = np.sum(black_img, axis=1)
    # print(sum_of_rows)

    lines = []

    NOT_FOUND_YET = -1
    start_y, end_y = -1, -1
    for y, sum in enumerate(sum_of_rows):
        if sum > 0:
            if start_y == -1:
                start_y = y
        else:  # sum == 0, nothing
            if start_y > -1:
                end_y = y
                lines.append((start_y, end_y))
                start_y = -1
                end_y = -1

    print(lines)

    
    orderedWords = [[] for _ in range(len(lines))]
    
    # line grouping
    for detection in detections:

        bl = detection[0]  # bottom-left
        tl = detection[1]
        tr = detection[2]
        br = detection[3]

        minv = min(tl[1], tr[1])
        maxv = max(bl[1], br[1])

        for i, line in enumerate(lines):
            if minv >= line[0] and maxv <= line[1]:
                orderedWords[i].append(detection)
    for row in orderedWords:
        print(row)

    print("BEFORE ^^^^^^^^^^^")

    # print(orderedWords)
    def custom_sort(contours):
        return [contour[0][0] for contour in contours]
    # [[x, y] [x, y] [x, y] [x, y]]
    for row in orderedWords:
        # predicate = 
        # row.sort(predicate)
        # just_first = custom_sort(row)
        # [25, 235 , 52142]
        # just_first.sort(axis=0)
        ordering = np.argsort(np.array(row), axis=0)
        print(ordering)
        print(np.array(row))
        print(row)
        np.take_along_axis(np.array(row), ordering, axis=0)
        # orderedWords = sorted(row, key=lambda detection: min(detection[0][0], detection[1][0]))
    
    # for row in orderedWords:
    #     print(row)
    # sum_of_rows = cv2.threshold(sum_of_rows, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    
    # plt.figure()
    # plt.imshow(black_img, cmap="gray")
    # plt.show()
    return orderedWords
    
if __name__ == "__main__":
    # main()
    recognizeText()


a = np.array([[1,4], [3,1]])

a.sort(axis=1)

a
array([[1, 4],
       [1, 3]])

a.sort(axis=0)

a
array([[1, 3],
       [1, 4]])
[[1, 4], [3, 1]] axis = 1

row = [[ [], [], [], [] ] [dim2] [dim3]]
dim = [[] [] [] []]
