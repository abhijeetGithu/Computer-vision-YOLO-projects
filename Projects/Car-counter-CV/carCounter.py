from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
model = YOLO('./YOLO -weights/yolov8n.pt')
# cap=cv2.VideoCapture(0)
cap = cv2.VideoCapture("./videos/cars.mp4")  # for videos
# cap.set(3,1280)
# cap.set(4,720)
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

mask = cv2.imread("mask.png")
#tracking cars
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limit=[400,297,673,297]
TotalCount=[]
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics=cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    img=cvzone.overlayPNG(img,imgGraphics,(0,0))
    result = model(imgRegion, stream=True)

    detections = np.empty((0, 5))


    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)\

            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == 'car' or currentClass == 'motorbike' or currentClass == 'truck' or currentClass == 'bus' and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,
                                   offset=4)
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=5)

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))


    resultTracker = tracker.update(detections)
    cv2.line(img,(limit[0],limit[1]),(limit[2],limit[3]),(0,0,255),5)

    for result in resultTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        #print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5,colorR=(255,0,255))


        # You can either put Car id on car boundry or no. of car counted
        # cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
        #                    scale=2, thickness=3, offset=10)

        cx,cy=x1+w//2,y1+h//2
        if limit[0] < cx < limit[2] and limit[1]-15 < cy < limit[1]+15:
            if TotalCount.count(id) == 0:
                TotalCount.append(id)
                cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0, 255,0), 5)

        # cvzone.putTextRect(img, f'Count:{len(TotalCount)}',(50,50))


    # Here we have use no. of car Counted to be displayed on the car boundry

    cvzone.putTextRect(img, f'{len(TotalCount)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
                              offset=10)
    cv2.putText(img, str(len(TotalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow('image', img)
    # cv2.imshow('imageRegion', imgRegion)
    cv2.waitKey(1)
