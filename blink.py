import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance

detector = dlib.get_frontal_face_detector()

PREDICTOR = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

BEFORE = [0, 0, 0, 0]

shape = []

I = 0

## If camera is slow startup, you choose bellow command.
CAM = cv2.VideoCapture(1)
# CAM = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#######

def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)

while 1:
    ret, frame = CAM.read()
    
    mono = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(mono, 1)

    for (i, rect) in enumerate(rects):
        shape = PREDICTOR(mono, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        j = 0
        for (x, y) in shape:
            j = j + 1
            if(j > 36 and j < 49):
                cv2.circle(frame, (x,y), 2, (0, 255, 0), -1)

        
    L_eye = calc_ear(shape[42:48])
    R_eye = calc_ear(shape[36:42])

    if L_eye < 0.2 or R_eye < 0.2:
        I += 1
    
    cv2.putText(frame, f"blink detect-{I}", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break


CAM.release()
cv2.destroyAllWindows()