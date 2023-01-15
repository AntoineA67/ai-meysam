import cv2

# read an image file
cap = cv2.VideoCapture('http://192.168.1.26:4747/mjpegfeed')

while True:
    ret, frame = cap.read()

    cv2.imshow("frame", frame)
    cv2.waitKey(1)