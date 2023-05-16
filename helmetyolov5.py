import torch
import cv2
import numpy as np
from tracker import Tracker

# configure trained model
model = torch.hub.load('C:/Users/knigh/DataspellProjects/Training/yolov5', 'custom',
                       path='C:/Users/knigh/DataspellProjects/Training/dnn_model/helmetweights.pt', source='local')
model.classes = [0, 1]  # specify classes

# specify video input
cap = cv2.VideoCapture("vidtest/testvid.mp4")

# count for tracker id
count = 0
tracker = Tracker()


# looking at the coordinates to configure the ROI
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


# set frame by frame
cv2.namedWindow('FRAME')
# cv2.setMouseCallback('FRAME', POINTS)

# set region of interest
roi = [(311, 222), (189, 276), (606, 459), (649, 338)]
region1 = set()

# detect object at the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    detect = model(frame)

    list = []
    for index, rows in detect.pandas().xyxy[0].iterrows():  # to get coordinates from the classes
        x = int(rows[0])
        y = int(rows[1])
        w = int(rows[2])
        h = int(rows[3])
        class_name = str(rows['name'])
        list.append([x, y, w, h])

    idx_bbox = tracker.update(list)  # put together the coordinates

    for bbox in idx_bbox:
        x, y, w, h, class_id = bbox
        cx = int((x + w) / 2)  # x center point coordinates
        cy = int((y + h) / 2)  # y center point coordinates
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
        cv2.putText(frame, str(class_id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

        if "Motorcycle" in class_name:
            inside_region = cv2.pointPolygonTest(np.array(roi, np.int32), (cx, cy), False)

            if inside_region > 0:
                region1.add(class_id)

    # display the roi
    cv2.polylines(frame, [np.array(roi, np.int32)], True, (0, 255, 0), 2)

    # display the count
    helmet_count = len(region1)
    cv2.putText(frame, "Pengguna Helm: " + str(helmet_count), (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # display the video
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(25) & 0xFF == 25:
        break

cap.release()
cv2.destroyAllWindows()
# %%
