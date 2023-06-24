import os
import torch
import cv2
import numpy as np
# from deep_sort import DeepSort
from tracker import Tracker
from datetime import datetime

# configure trained model
model = torch.hub.load('yolov5', 'custom', path='dnn_model/helmetweightsv4.pt', source='local')
model.classes = [0]  # specify classes

# specify video input
cap = cv2.VideoCapture("vidtest/cctv1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frmDelay = int(1000 / fps)

# count for tracker id
count = 0
tracker = Tracker()


# looking at the coordinates to configure the ROI
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


# set date for captured image
def imgWrite(img):
    now = datetime.now()
    currentTime = now.strftime("%H_%M_%S_%d_%m_%Y")
    filename = "%s.png" % currentTime
    cv2.imwrite(os.path.join(r"C:\Users\knigh\DataspellProjects\Training\capturedData\CCTV1", filename), img)


# set frame by frame
cv2.namedWindow('FRAME')
# cv2.setMouseCallback('FRAME', POINTS)

# set region of interest
roi = [(267, 75), (162, 117), (579, 306), (628, 300), (640, 201)]
roi2 = [(10, 150), (10, 460), (656, 460), (656, 150)]
helmetDetect = set()
noHelmetDetect = set()

# detect object at the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (852, 480))
    droi = frame[150:460, 10:670]
    detect = model(droi)

    list = []
    for index, rows in detect.pandas().xyxy[0].iterrows():  # to get coordinates from the classes
        x = int(rows[0])
        y = int(rows[1])
        w = int(rows[2])
        h = int(rows[3])
        class_name = str(rows['name'])
        list.append([x, y, w, h])
        # print(detect.pandas().xyxy[0])

    idx_bbox = tracker.update(list)  # put together the coordinates

    for bbox in idx_bbox:
        x, y, w, h, class_id = bbox
        cx = int((x + w) / 2)  # x center point coordinates
        cy = int((y + h) / 2)  # y center point coordinates
        cv2.rectangle(droi, (x, y), (w, h), (0, 0, 255), 2)
        cv2.putText(droi, str(class_id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.circle(droi, (cx, cy), 2, (0, 255, 0), -1)

        if "Helmet" in class_name:
            inside_region = cv2.pointPolygonTest(np.array(roi, np.int32), (cx, cy), False)

            if inside_region > 0:
                helmetDetect.add(class_id)

        if "No-Helmet" in class_name:
            inside_region = cv2.pointPolygonTest(np.array(roi, np.int32), (cx, cy), False)

            if inside_region > 0:
                noHelmetDetect.add(class_id)
                crop = droi[y:h, x:w]
                imgWrite(crop)

    # display the roi
    cv2.polylines(droi, [np.array(roi, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(roi2, np.int32)], True, (255, 255, 0), 2)

    # display the count
    helmetCount = len(helmetDetect)
    noHelmetCount = len(noHelmetDetect)
    cv2.putText(frame, "Menggunakan Helm: " + str(helmetCount), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                2)
    cv2.putText(frame, "Tidak Menggunakan Helm: " + str(noHelmetCount), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)

    # display the video
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#%%
