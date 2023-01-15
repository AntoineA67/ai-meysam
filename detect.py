import cv2
import time
import requests

# OpenCV's deep learning face detector
net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

# net = cv2.dnn_TextDetectionModel_DB("models/DB_IC15_resnet50.onnx")
# set post-processing params
# net.setBinaryThreshold(0.3)
# net.setPolygonThreshold(0.5)
# net.setMaxCandidates(200)
# net.setUnclipRatio(2.0)

# # set input shape and normalization params
# net.setInputScale(1.0 / 255.0)
# net.setInputSize(736, 736)
# net.setInputMean((122.67891434, 116.66876762, 104.00698793))

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ROBOT_URL = 'http://127.0.0.1:5000/'

# open a video capturer
cap = cv2.VideoCapture("http://192.168.1.26:4747/mjpegfeed")

startTime = time.time()

while True:
    # read a frame
    ret, frame = cap.read()

    frame = cv2.rotate(frame, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)

    # check if the frame was captured correctly
    if not ret:
        break

    # resize the frame and convert to a blob
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and get the detections
    # rotated_rectangles, _ = net.detectTextRectangles(frame)

    # for rotated_rectangle in rotated_rectangles:
        # print(type(rotated_rectangle))
        # print(rotated_rectangle)
        # points = cv2.boxPoints(rotated_rectangle) # error in opencv 4.5.4
        # print(points)
        # if len(points) > 0:
        # pt1 = (int(rotated_rectangle[0][0]), int(rotated_rectangle[0][1]))
        # pt2 = (int(rotated_rectangle[1][0]), int(rotated_rectangle[1][1]))
        # cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

    net.setInput(blob)
    detections = net.forward()

    # loop over the detections

    x2, x1, y2, y1 = frame.shape[1], 0, 0, 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence < 0.5:
            continue

        # get the x, y coordinates of the bounding box
        x1 = int(detections[0, 0, i, 3] * frame.shape[1])
        y1 = int(detections[0, 0, i, 4] * frame.shape[0])
        x2 = int(detections[0, 0, i, 5] * frame.shape[1])
        y2 = int(detections[0, 0, i, 6] * frame.shape[0])

        size = (x2 - x1) * (y2 - y1)

        # draw the bounding box and display the coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "x: {}, y: {}".format(x1, y1), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    current_time = time.time()
    elapsed_time = current_time - startTime
    if elapsed_time >= 0.2:

        startTime = current_time

        x = (x1 + x2 - frame.shape[1]) // 2 // ((x2 - x1) // 20)

        params = {'x': str(x)}
        print(x1, x2, params)

        response = requests.get(ROBOT_URL + 'stop')
        response = requests.get(ROBOT_URL + 'custom', params=params)
        

    # show the frame
    cv2.imshow("Frame", frame)

    # wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        requests.get(ROBOT_URL + 'stop')
        break

# release the capturer
cap.release()

# close all windows
cv2.destroyAllWindows()
