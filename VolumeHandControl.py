import cv2 as cv
import mediapipe as mp
import time 
import HandTrackingModule as htm
import numpy as np
import math

#for screen brightness control
import screen_brightness_control as sbc

# Importing necessary libraries for controlling system volume 
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities
from pycaw.pycaw import IAudioEndpointVolume

#Defining the width and height of the window
wCam, hCam = 640,480


cap = cv.VideoCapture(0)
cap.set(3, wCam)  # Width
cap.set(4, hCam)  # Height
ptime = 0
detector = htm.handDetector(detectionConfi=0.7)

#Volume control setup
device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# print(f"Audio output: {device.FriendlyName}")
# print(f"- Muted: {bool(volume.GetMute())}")
volRange  = volume.GetVolumeRange()
# print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
volume.SetMasterVolumeLevel(0, None)

minVol = volRange[0]  # Minimum volume level
maxVol = volRange[1]  # Maximum volume level
vol = 0  # Initial volume level
volBar = 400 # Initial volume bar height

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture image")
        break
    
    # Find hands
    frame = detector.findHands(frame, draw = True)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        """
        Getting the position of the index finger and thumb tips 
        and calculating the distance between them to control volume
        index finger tip is at index 8 and thumb tip is at index 4
        """
        # Get the tip of the index finger
        index_finger_tip = lmList[8][1:3]
        # Get the tip of the thumb
        thumb_tip = lmList[4][1:3]
        
        #these vars contain the x and y coordinates of the index finger and thumb tips
        x1, y1 = index_finger_tip
        x2, y2 = thumb_tip
        #getting the center of the line connecting the index finger and thumb tips
        cx , cy = (x1+x2)//2, (y1+y2)//2
        
        
        #verify if the fingers are correctly detected
        cv.circle(frame, (x1, y1), 10, (255, 0, 255), cv.FILLED)
        cv.circle(frame, (x2, y2), 10, (255, 0, 255), cv.FILLED)
        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv.circle(frame, (cx,cy),10, (255, 0 , 255), cv.FILLED)  # Draw a circle at the center of the line
        
        
        length = math.hypot(x2-x1, y2-y1)  # Calculate the distance between the index finger and thumb tips
        
        
        #hand range : 50-300
        #Volume range : -65.25 to 0.0
        
        #converting the hand range to a volume range
        vol = np.interp(length, [80,250], [minVol, maxVol])
        volBar = np.interp(length, [80,250], [400, 150])  # Map length to volume bar height
        brightness = np.interp(length, [80,250], [0, 100])  # Map length to brightness range (0 to 100)
        print(f"Volume: {vol:.2f} dB and Brightness: {brightness:.2f}%")
        
        #setting the system brightness level as brightness
        sbc.set_brightness(int(brightness))
        
        #setting the system volume level as vol
        volume.SetMasterVolumeLevel(vol, None)
        
        if length<10:
            cv.circle(frame , (cx,cy),10,(0,255,0), cv.FILLED)  # Draw a filled circle at the center of the line if fingers are close
        
    
    cv.rectangle(frame, (50, 150), (85,400), (0, 255, 0), 3)  # Draw a rectangle border for volume display
    #display the volume level in the rectangle
    cv.rectangle(frame, (50, int(volBar)),(85,400), (0, 255, 0), cv.FILLED)  # Fill the rectangle with green color
    cv.putText(frame, f'Volume: {int((vol - minVol) / (maxVol - minVol) * 100)}%', (60, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    #fps
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(frame, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    
    """
    This works but is not used in the final code
    
    
    frame = detector.findHands(frame, draw=True)
    lmList = detector.findPosition(frame, draw=False)
    
    if len(lmList) != 0:
        # Get the tip of the index finger
        index_finger_tip = lmList[8][1:3]
        
        # Get the tip of the thumb
        thumb_tip = lmList[4][1:3]
        
        # Calculate the distance between the index finger and thumb tips
        distance = np.linalg.norm(np.array(index_finger_tip) - np.array(thumb_tip))
        
        # Map the distance to a volume level (0 to 100)
        volume_level = int(np.clip(distance, 0, 300) / 3)
        
        # Display the volume level on the frame
        cv.putText(frame, f'Volume Level: {volume_level}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    """
    
    
    cv.imshow("Volume Hand Control", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
