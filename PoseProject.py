import cv2 as cv
import time
import PoseModule as pm

pTime = 0
cap = cv.VideoCapture('PoseVideos/1.mp4')

detector = pm.poseDetector()


while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList= detector.getPosition(img)

    if len(lmList)!=0:
        print(lmList[14])
        # cv.circle(img, (lmList[14][1], lmList[14][2]), 5, (112, 200, 150), cv.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3,
                (199, 90, 100), 2)

    cv.imshow("Image", img)
    cv.waitKey(1)