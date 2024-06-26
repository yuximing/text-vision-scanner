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
    from openai import OpenAI
    from dotenv import load_dotenv
    import os
    # from openAIConnect import OpenAI
    load_dotenv()
    client = OpenAI(
        api_key = os.environ.get("OPENAI_API_KEY")
    )
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


    
    words = orderWords(detections, image.shape)
    words_arr = []
    for line in words:
        for detection in line:
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
            # print(word)
            words_arr.append(word)
        # print()

    concat_str = " ".join(words_arr)
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "you are a nerdy assistant, answer with big nerdy words"},
        {"role": "user", "content": f"tell me what the following text is talking about: {concat_str}"}
    ]
    )

    print(completion.choices[0].message.content)

def orderWords(detections, imageDimensions):
    # display the contours on a black bg

    # Create black background
    black_img = np.zeros((imageDimensions[0], imageDimensions[1]), dtype=np.int32)
    black_img = cv2.polylines(black_img, detections, isClosed=True, color=(255,), thickness=1)
    sum_of_rows = np.sum(black_img, axis=1)

    # DO NOT DELETE ME
    # [ 
    #     [ [0] [0] [0] [0]], black_img[0] =>  [ [0] [255] [0] [255]]  [  x  x]    say 3-d, [[255, 255, 255] [255, 255, 255] [0, 255, 255]] = [[255 *2+0, 255 *3, 255 *3]]
    #     [ [0] [0] [0] [0]],                  [ [255] [255] [255] [255]]  [  x  x]
    #     [ [0] [0] [0] [0]],
    #     [ [] [] [] []],
    # ]

    lines = []

    start_y, end_y = -1, -1  # -1 means that y hasn't been selected yet
    for y, sum in enumerate(sum_of_rows):
        if sum > 0:  # detected contour
            if start_y == -1:
                start_y = y
        else:  # no contour
            if start_y > -1:
                end_y = y
                lines.append((start_y, end_y))
                start_y, end_y = -1, -1

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

    for i, row in enumerate(orderedWords):
        orderedWords[i] = sorted(row, key=lambda mat: mat[0][0])
    # # # print the rows
    # for row in orderedWords:
    #     print(row)

    sum_of_rows = sum_of_rows.astype(np.uint8)
    sum_of_rows = np.expand_dims(sum_of_rows, axis=1)
    sum_of_rows = np.clip(sum_of_rows, 0, 255)
    _, sum_of_rows = cv2.threshold(sum_of_rows, 1, 255, cv2.THRESH_BINARY)

    # ax1 = plt.subplot(121)
    # plt.imshow(black_img, cmap='gray', aspect='auto')
    # ax2 = plt.subplot(122)
    # plt.imshow(sum_of_rows, cmap="gray", aspect='auto')
    # plt.show()

    return orderedWords
    
if __name__ == "__main__":
    # main()
    recognizeText()


# a = np.array([[1,4], [3,1]])

# a.sort(axis=1)

# a
# array([[1, 4],
#        [1, 3]])

# a.sort(axis=0)

# a
# array([[1, 3],
#        [1, 4]])
# [[1, 4], [3, 1]] axis = 1

# row = [[ [], [], [], [] ] [dim2] [dim3]]
# dim = [[] [] [] []]
