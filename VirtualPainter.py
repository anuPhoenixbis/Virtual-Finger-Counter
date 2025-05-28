import cv2 as cv
import mediapipe as mp
import time
import os
import HandTrackingModule as htm
import numpy as np

image_folder = r'D:\VS_Code_WorkSpaces\OpenCV_Personal\Hand_Tracking\PNG'
image_path = os.listdir(image_folder)
# print(image_path)
overlayList = []
for imPath in image_path:
    image = cv.imread(f'{image_folder}/{imPath}')
    image = cv.resize(image , (1280,125))
    overlayList.append(image)
# print(len(overlayList))
header= overlayList[0]
drawColor = (255,0,255)

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

brushThickness=15
eraserThickness = 100

#prev coords
xp , yp =0,0
imgCanvas = np.zeros ((720,1280,3) , np.uint8)

detector = htm.handDetector(detectionConfi = .85)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv.flip(frame , 1) #to make the video in the correct orientation
    
    """
    1. Import images
    2. Find hand landmarks
    3. Check the number of fingers are up
    4. If Selection Mode - 2 fingers are up
    5. If Drawing Mode - 1 finger is up
    """
    
    frame = detector.findHands(frame , draw = True)
    lmList = detector.findPosition(frame , draw=False)
    
    if len(lmList) != 0:
        # print(lmList)
        
        
        x1,y1 = lmList[8][1:] #coords of the tip of the index finger
        x2,y2 = lmList[12][1:] #coords of the tip of the middle finger
    
    
        fingers = detector.fingersUp()
        # print(fingers)
        
        
        #Selection Mode
        if fingers[1] and fingers[2]:
            # print('Selection Mode')
            #whenever we select a new color or eraser update the coords
            xp, yp =0,0
            #selecting colors
            if y1<125:#this means we are at the top of the screen (header)
                #selection for the color
                if 250<x1<450 :
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550<x1<750 : 
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 800<x1<950 :
                    header = overlayList[2]
                    drawColor = (158,8,3)
                elif 1050<x1<1200 :
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv.rectangle(frame , (x1,y1-15) , (x2,y2+15) , drawColor , cv.FILLED)
        
        #Drawing mode
        if fingers[1] and not fingers[2]:
            cv.circle(frame , (x1,y1) , 15, drawColor , cv.FILLED)
            # print('Drawing Mode')
            #determining the prev coords
            #for the very first frame
            if xp == 0 and yp == 0:
                xp,yp = x1,y1
            
            if drawColor == (0,0,0):
                cv.line(frame , (xp,yp) , (x1,y1) , drawColor , eraserThickness)
                #we shall draw on the imgCanvas instead of the original frame
                cv.line(imgCanvas , (xp,yp) , (x1,y1) , drawColor , eraserThickness)
            else : 
                #drawing a line b/w the curr coords and the prev coords
                cv.line(frame , (xp,yp) , (x1,y1) , drawColor , brushThickness)
                #we shall draw on the imgCanvas instead of the original frame
                cv.line(imgCanvas , (xp,yp) , (x1,y1) , drawColor , brushThickness)

            
            #updating the prev coords
            xp,yp = x1,y1
    
    #merging the canvas and original frame
    imgGray = cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    _,imgInv = cv.threshold(imgGray , 50,255,cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    frame = cv.bitwise_and(frame,imgInv)
    frame = cv.bitwise_or(frame,imgCanvas)
    
    
    #displaying the images as the header
    frame[0:125 , 0:1280] = header
    
    # frame = cv.addWeighted(frame,.5,imgCanvas,.5,0)
    
    cv.imshow("Video" , frame)
    # cv.imshow("Canvas" , imgCanvas)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    