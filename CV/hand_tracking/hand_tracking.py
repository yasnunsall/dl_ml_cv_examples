import cv2
import time
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hand = mp.solutions.hands

hands = mp_hand.Hands()

mp_draw = mp.solutions.drawing_utils

p_time = 0
c_time = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1200, 800), interpolation = cv2.INTER_AREA)
    print(img.shape)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)
    print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms,
                                   mp_hand.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                
                #bilek
                if id == 4:
                    cv2.circle(img, (cx, cy), 9, (255, 0, 0), cv2.FILLED)
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    
    
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 75),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)
    
    cv2.imshow("img", img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
            
    
cap.release()
cv2.destroyAllWindows()