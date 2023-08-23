import cv2
import mediapipe as mp
import time

# read camera 
cap = cv2.VideoCapture(0)


# making an instance from mpHands
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# using mp Draw module that app would be able to draw landmarks on iamge
mpDraw = mp.solutions.drawing_utils

while True:
    isCaptureSuccessful,image = cap.read()
    
    if isCaptureSuccessful:
        
        # changing image color's channel to RGB so mediapipe can read image
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        # calling hand detection process function 
        results = hands.process(imageRGB)

        # check if any hands detected or not
        if results.multi_hand_landmarks:

            # unpacking hands landmarks
            for handLms in results.multi_hand_landmarks:
                # unpacking each hand's landmarks
                for id,lm in enumerate(handLms.landmark):

                    # getting image height,width,color_channels for changing decimal lm places
                    #  on the image to x,y integer pixel values after 
                    image_height,image_width,image_channels = image.shape

                    # chnaging decimal values to integer pixel coordinates on image multiplication
                    cx,cy = int(lm.x*image_width),int(lm.y*image_height)
                    print(id,cx,cy)

                # drawing each hand landmarks on original image using precessed values
                mpDraw.draw_landmarks(image,handLms,mpHands.HAND_CONNECTIONS)
        
        # displaying outputs
        cv2.imshow("Image" , image)
    
    # waiting 1 ms so check if user pressed 'q' button to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
