from ultralytics import YOLO
import cv2
import cvzone
import math
import imutils
# initialize the cv2 QRCode detector
detector = cv2.QRCodeDetector()
# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture(0)  # For Video
model = YOLO("/ObjectDetection/QRNEW.pt")

classNames = ['QR_CODE']
myColor = (0, 0, 255)
while True:
    success, img = cap.read()
    img = imutils.resize(img, width=640, height=384)
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
            
    cv2.imshow("Image",img)
    cv2.waitKey(1)