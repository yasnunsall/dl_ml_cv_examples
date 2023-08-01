import mediapipe as mp
import cv2 

cap = cv2.VideoCapture(0)

mp_hand= mp.solutions.hands
hands = mp_hand.Hands()
mp_draw = mp.solutions.drawing_utils

tip_ids = [4, 8, 12, 16, 20]

while True:
    succes, img = cap.read()
    img = cv2.resize(img, (1000, 700), cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)
    
    lm_list = []
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms,
                                   mp_hand.HAND_CONNECTIONS)
            
            for id, lm in enumerate(hand_lms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_list.append([id, cx, cy])
                
                #isaret uc = 8
                if id == 8:
                    cv2.circle(img, (cx, cy), 9, (255, 0, 0), cv2.FILLED)
    
    if len(lm_list) != 0:
        fingers = []
        
            
        for id in range(0, 5):
            if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        total_f = fingers.count(1)
        
        cv2.putText(img, str(total_f), (30, 125), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 8)
        
    
    cv2.imshow("img", img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()