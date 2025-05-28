import cv2 as cv
import mediapipe as mp
import HandTrackingModule as htm
import numpy as np
import autopy
import time

wCam = 640
hCam = 480
frameR = 100 #frame reduction

cap = cv.VideoCapture(0)
pTime = 0
prevlocX , prevlocY =0,0
currlocX , currlocY = 0,0
cap.set(3,wCam)
cap.set(4,hCam)
detector = htm.handDetector(maxhands=1)
wScr , hScr = autopy.screen.size()
smoothening = 7

while cap.isOpened():
    success , frame = cap.read()
    if not success:
        break
    """
    Stepwise breakdown:
    1. find landmarks
    2. get the tip of the index and middle finger
    3. Check which are up
    4. Moving mode : only index up
    5. convert coords
    6. Smoothen values
    7. Move mouse
    8. Clicking mode : index and middle are up
    10. Click mouse if dist is short
    11. Frame rate
    """
    
    #finding lms
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)
    
    #get the tip of the index and middle finger
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:3]
        x2,y2 = lmList[12][1:3]
        
        #check which fingers are up
        fingers = detector.fingersUp()
        
        #this box denotes the interpreted screen of the pc in the video
        cv.rectangle(frame,(frameR,frameR) , (wCam-frameR,hCam-frameR) , (255,0,255) , 2)

        
        #moving mode
        if fingers[1]==1 and fingers[2]==0:
            print("Moving mode")
        
            #convert coords
            x3 = np.interp(x1 , (frameR,wCam-frameR) , (0,wScr))
            y3 = np.interp(y1 , (frameR,hCam-frameR) , (0,hScr))
            
            
            #smoothen the values to avoid shaking
            currlocX = prevlocX + (x3-prevlocX) /smoothening
            currloY = prevlocY + (y3-prevlocY) /smoothening
            
            
            #sending the values to the mouse
            autopy.mouse.move(wScr-currlocX,currlocY)
            cv.circle(frame,(x1,y1) , 15 , (255,0,255) , cv.FILLED)
            
            #updation
            prevlocX , prevlocY = currlocX , currlocY
        
        #clicking mode
        if fingers[1]==1 and fingers[2]==1:
            length , frame , lineInfo = detector.findDistance(8,12,frame)
            
            #if the distance falls below this threshold value then it means clicking
            if length<40:
                #when clicked the circle turns green
                cv.circle(frame,(lineInfo[4],lineInfo[5]) , 15 , (0,255,0) , cv.FILLED)
                #actual clicking
                autopy.mouse.click()

    
    
    #fps calc
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv.putText(frame , f'FPS : {str(int(fps))}' , (20,50) , cv.FONT_HERSHEY_PLAIN , 3 ,(255,0,0) , 2)
    
    cv.imshow("Video" , frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break