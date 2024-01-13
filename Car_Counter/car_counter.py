from ultralytics import YOLO
import imutils
import cv2
import cvzone
import math
from sort import *    # download sort and import it
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("D:\objectdetection_yolo\pythonProject2\ObjectDetection\pexels-tim-samuel-5834623 (2160p).mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
model = YOLO('yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# mask = cv2.imread('mask-car-counter (2).png')
#tracking
tracker = Sort(max_age = 20,min_hits = 3,iou_threshold=0.3)   #max_age = number of frames to change
limits = [220,150,480,150]
totalCounts = []
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640,height= 384)
    # imageRegion = cv2.bitwise_and(frame,mask)
    if ret == True:
       result = model(frame,stream=True)

       detections = np.empty((0,5))
       for r in result:
           boxes = r.boxes
           for box in boxes:
               #bounding box
               x1,y1,x2,y2 = box.xyxy[0]
               x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)

               #Confidence
               conf = math.ceil((box.conf[0]*100))/100

               #Class Name
               cls = int(box.cls[0])
               currentClass = classNames[cls]
               if(currentClass == 'car' or currentClass == 'truck' or currentClass == 'bicycle' and conf>0.4):
                   # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
                   # cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(30, y1)),thickness=1,scale=1,offset=2)
                   currentArray = np.array([x1,y1,x2,y2,conf])
                   detections = np.vstack((detections,currentArray))
       resultsTracker = tracker.update(detections)
       cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),color=(0,225,0),thickness=3)
       for results in resultsTracker:
           x1, y1, x2, y2, id = results
           x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
           print(results)
           w,h = x2-x1,y2-y1
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
           cvzone.putTextRect(frame, f'{int(id)} ', (max(0, x1), max(30, y1)), thickness=1, scale=1,
                              offset=2)
           cx,cy = x1+w//2,y1+h//2
           cv2.circle(frame,(cx,cy),5,(0,225,0),cv2.FILLED)

           if limits[0]<cx<limits[2] and limits[1]-10<cy<limits[1]+10:
               if totalCounts.count(id) ==0:
                   totalCounts.append(id)
                   cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 225, 0), thickness=3)

       cvzone.putTextRect(frame,str(len(totalCounts)),(255,100),2,offset = 4)
       cv2.imshow('frame', frame)



       if cv2.waitKey(1) & 0xFF == ord('q'):
         break
    else:
        break

cap.release()
cv2.destroyAllWindows()