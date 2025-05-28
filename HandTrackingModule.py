import cv2 as cv
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self ,  mode=False, maxhands =2 , detectionConfi = .5 , trackingConfi=.5):
        
        self.mode=mode
        self.maxhands=maxhands
        self.detectionConfi=detectionConfi
        self.trackingConfi=trackingConfi

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, max_num_hands=self.maxhands, min_detection_confidence=self.detectionConfi , min_tracking_confidence=self.trackingConfi) 
        self.mpDrawing = mp.solutions.drawing_utils #for drawing the landmarks on the image
        self.tipIds=[4,8,12,16,20]  # List of tip IDs for fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3): #to get the distance b/w finger tips 
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
    
    
    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB) #converting the image to RGB format
        self.results = self.hands.process(imgRGB) #processing the image to detect hands
    

        if self.results.multi_hand_landmarks: #if hands are detected
            for handLms in self.results.multi_hand_landmarks: #iterating through each hand detected
                if draw: #if draw is True, draw the landmarks on the image
                    self.mpDrawing.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS) #drawing the landmarks on the image
        return img


    def findPosition(self, img, handNo=0, draw=True):
        
        self.lmList = [] #list to store the landmarks positions
        
        if self.results.multi_hand_landmarks: #if hands are detected
            
            myHand = self.results.multi_hand_landmarks[handNo] #getting the hand landmarks of the specified hand (handNo)
            for id,landmark in enumerate(myHand.landmark): #iterating through each landmark of the hand
                
                #id is the index of each landmark; landmark is the landmark object containing x, y, z coordinates(its the points on the hand)
                h, w, c = img.shape #getting the height, width and channels of the image; these are ratios of the original image
                
                #landmark.x and landmark.y are normalized coordinates (0 to 1) of the landmark
                cx, cy = int(landmark.x * w), int(landmark.y * h) #calculating the x, y coordinates of the landmark
                
                self.lmList.append([id, cx , cy]) #appending the landmark id and its coordinates to the list
                
                if draw: #if draw is True, draw the landmarks on the image
                    cv.circle(img, (cx,cy), 5, (0,0,255), cv.FILLED) #draw a filled circle at the landmark position with a smaller radius
        return self.lmList #returning the list of landmarks positions
                    #     """
                    #     Making one of the landmark special, in this case the wrist (id=0)
                    #     Drawing a large circle at the wrist pos and smaller circles at the others
                    #     This way we can easily identify the different landmarks
                    #     """
                    #     if id==0: #if the landmark is the wrist (id=0)
                    #         cv.circle(img, (cx,cy), 15, (255,0,255), cv.FILLED) #draw a filled circle at the wrist position
                    #     else: #for other landmarks
                    #         cv.circle(img, (cx,cy), 10, (0,255,0), cv.FILLED)
                    #         #draw a filled circle at the landmark position with a smaller radius
        
    def fingersUp(self):
        fingers = []
    
        #for the thumbs 
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)  # Thumb is up
            # cv.putText(frame, 'Thumb Up', (400, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            fingers.append(0)
        """
        if the tip of any finger is above the point just above the knuckle point, it counts as a finger up 
        """
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
                # cv.putText(frame, f'Finger {id} Up', (400, 50 + id * 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                fingers.append(0)  
        return fingers      


def main():
    
    pTime = 0
    currTime = 0 
    cap = cv.VideoCapture(0) #giving the camera input
    
    # initialize the handDetector class
    detector = handDetector()   
    
    while True:
        success, img = cap.read() #reading the camera input
        if not success:
            print("Failed to capture image")
            break
                
        img = detector.findHands(img) #detecting hands in the image
        lmList = detector.findPosition(img) #getting the landmarks positions
        
        if len(lmList) != 0: #if landmarks are detected
            print(lmList[4]) #printing the position of the 5th landmark (index 4), which is the thumb tip
        
        #calculating the frame rate(FPS)
        currTime = time.time() #getting the current time
        fps = 1/(currTime - pTime)
        pTime = currTime #updating the previous time variable
        #display the FPS on the image
        cv.putText(img, f'FPS : {int(fps)}', (10,30) , cv.FONT_HERSHEY_SIMPLEX , 1, (0,255,0), 2, cv.LINE_AA)
        #This line draws the FPS value as green text at position (10, 30) on the image, using a simple font, size 1, thickness 2, and anti-aliased edges.
        
        cv.imshow("Image", img) #showing the image captured by camera
        if cv.waitKey(1) & 0xFF == ord('q'): #if 'q' is pressed, break the loop
            break
        time.sleep(0.1) #adding a small delay to avoid high CPU usage




if __name__ == "__main__":
    main()