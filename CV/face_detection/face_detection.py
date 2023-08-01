import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(.20)

mp_draw = mp.solutions.drawing_utils

while True:
    succes, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = face_detection.process(img_rgb)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            bbox_c = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            bbox = int(bbox_c.xmin*w), int(bbox_c.ymin*h), int(bbox_c.width*w), int(bbox_c.height*h)
            cv2.rectangle(img, bbox, (0, 255, 255), 2)
            
    cv2.imshow("img", img)
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()