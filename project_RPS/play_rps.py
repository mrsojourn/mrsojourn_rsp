import os
import time
import random
import cv2
import mediapipe as mp
from utils_rps.HandTrackingModule import HandDetector
from utils_rps.FaceDetectionModule import FaceDetector

# Initialize hand detector and face detector
detectorH = HandDetector(maxHands=2, detectionCon=0.8)
detectorF = FaceDetector(minDetectionCon=0.5, modelSelection=1)

# Set video capture properties
w, h = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, w)
cap.set(4, h)

# Load Rock-Paper-Scissors images
folderPath = 'RPS'
myList = os.listdir(folderPath)
rpc_img = []
for p in myList:
    img = cv2.imread(f'{folderPath}/{p}')
    rpc_img.append(img)

gesture = ' '
result = ' '

start = False
timer = 0
count_time = False
num = 0

while True:
    _, img = cap.read()
    hc, wc, c = rpc_img[0].shape

    # Find hands in the image
    img = detectorH.findHands(img)
    lmList, bbox = detectorH.findPosition(img)

    # Find faces in the image
    img, bboxsFace = detectorF.findFaces(img)

    if start:
        if count_time is False:
            timer = time.time() - initialTime
            cv2.putText(img, str(int(timer)), (300, 300), cv2.FONT_HERSHEY_PLAIN, 8, (255, 255, 255), 4)

            if timer > 3:
                count_time = True
                timer = 0

                if bboxsFace:
                    x, y, w, h = bboxsFace[0]['bbox']
                    if bbox and detectorH.handType() == 'Left':
                        cx, cy = bbox['center']
                        fingers = detectorH.fingersUp()
                        if fingers == [1, 1, 1, 1, 1]:
                            gesture = 'Paper'
                        elif fingers == [0, 0, 0, 0, 0]:
                            gesture = 'Rock'
                        elif fingers == [0, 1, 1, 0, 0]:
                            gesture = 'Scissor'

                        num = random.randint(0, 2)

                        if (gesture == 'Paper' and num == 1) or (gesture == 'Rock' and num == 2) or (gesture == 'Scissor' and num == 0):
                            result = 'You WIN'
                        elif (gesture == 'Paper' and num == 0) or (gesture == 'Rock' and num == 1) or (gesture == 'Scissor' and num == 2):
                            result = 'DRAW'
                        else:
                            result = 'You LOSE'

                        cv2.putText(img, f'{result}'.upper(),(135, 150),cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 4)
                        cv2.putText(img, f'{gesture}'.upper(),(115, 150),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    if count_time:
        img[0:hc, 0:wc] = rpc_img[num]
        cv2.putText(img, f'{result}'.upper(), (135, 150), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 4)

    cv2.putText(img, 'Press \'s\' to start. Use left hand'.upper(), (150, 450), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

    cv2.imshow('window', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        start = True
        initialTime = time.time()
        count_time = False

cv2.destroyAllWindows()