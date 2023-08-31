import cv2 as cv
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, model_comp=1, smooth_lm=True,
                 enbl_segm=False,smooth_seg=True,  min_det=0.5,
                 min_track=0.5):
        self.mode=mode
        self.model_comp=model_comp
        self.smooth_lm=smooth_lm
        self.enbl_segm=enbl_segm
        self.smooth_seg=smooth_seg
        self.min_det=min_det
        self.min_track=min_track

        self.mpDraw= mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.model_comp,self.smooth_lm,
                                     self.enbl_segm,self.smooth_seg,
                                     self.min_det, self.min_track)

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx,cy), 5, (112, 200, 150), cv.FILLED)

        return lmList


def main():
    pTime = 0
    cap = cv.VideoCapture('PoseVideos/5.mp4')

    detector = poseDetector()


    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList= detector.getPosition(img)

        if len(lmList)!=0:
            print(lmList[14])
            cv.circle(img, (lmList[14][1], lmList[14][2]), 8, (190, 230, 150), cv.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3,
                   (199, 90, 100), 2)

        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()