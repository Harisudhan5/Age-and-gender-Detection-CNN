import cv2
from deepface import DeepFace
from mtcnn.mtcnn import MTCNN

mtcnn_detector = MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = mtcnn_detector.detect_faces(frame)
    if not faces:
        continue
    for face_info in faces:
        x, y, w, h = face_info['box']
        face = frame[y:y + h, x:x + w]

        if face.size != 0:
            result = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)
            age = int(result[0]['age'])
            gender = result[0]['gender']
            if gender['Woman'] > gender['Man']:gender = "Female"
            else:gender = "Male"
            print(gender)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Age: {age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 215, 0), 2)
            cv2.putText(frame, f'Gender: {gender}', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 215, 0), 2)

    cv2.imshow('Real-time Face, Age, Gender Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
