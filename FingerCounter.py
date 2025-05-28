import cv2 as cv
import mediapipe as mp
import time 
import HandTrackingModule as htm
import os

wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)  # Width
cap.set(4, hCam)  # Height
pTime = 0
detector = htm.handDetector(detectionConfi=0.75)
imagesPath = r"D:\VS_Code_WorkSpaces\OpenCV_Personal\Hand_Tracking\Finger_Photos"
myList = os.listdir(imagesPath)
# Load images and create a dictionary for finger count images
overlayList = []
for imgPath in myList:
    image = cv.imread(f'{imagesPath}/{imgPath}')
    resized_image = cv.resize(image, (200, 200))  # Resize to fit the overlay area
    overlayList.append(resized_image)
    
tipIds=[4,8,12,16,20]  # List of tip IDs for fingers

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    
    if len(lmList) != 0:
        fingers = []
        
        #for the thumbs 
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)  # Thumb is up
            # cv.putText(frame, 'Thumb Up', (400, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            fingers.append(0)
        """
        if the tip of any finger is above the point just above the knuckle point, it counts as a finger up 
        """
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
                # cv.putText(frame, f'Finger {id} Up', (400, 50 + id * 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)

        #adding the images to the frame
        frame[50:250, 0:200] = overlayList[totalFingers-1]  # Display the first image in the top-left corner
        
        cv.rectangle(frame , (20,275), (170, 450), (0, 255, 0), cv.FILLED)
        cv.putText(frame, str(totalFingers), (45, 425), cv.FONT_HERSHEY_SIMPLEX, 6, (255, 0 , 0), 25)
        
        
    #fps calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(frame, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Video", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break