import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    isCaptureSuccessful,image = cap.read()
    
    if isCaptureSuccessful:
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id,lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    image_height,image_width,image_channels = image.shape

                    cx,cy = int(lm.x*image_width),int(lm.y*image_height)
                    print(id,cx,cy)

                mpDraw.draw_landmarks(image,handLms,mpHands.HAND_CONNECTIONS)

        cv2.imshow("Image" , image)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
