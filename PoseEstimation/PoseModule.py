import cv2
import mediapipe as mp 
import time


class poseDetector():
    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackingCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth)
        


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
    
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
            
    def getPosition(self, img, draw=True):  
        lmLists = []
        if self.results.pose_landmarks:    
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmLists.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmLists
        
def main():
    cap = cv2.VideoCapture('p1.mp4')
    pTime = 0
    detector = poseDetector()
    new_width = 640  
    new_height = 480
    
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (new_width, new_height))   
        img = detector.findPose(img)    
        lmList = detector.getPosition(img)
        print(lmList[14])
                
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
            
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
    
if __name__ == "__main__":
    main()