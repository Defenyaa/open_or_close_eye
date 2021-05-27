from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np

def eyeAspectRatio(eye):
    # vertical
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # horizontal
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


earThresh = 0.15  # 闭眼阈值
incontinuity = 5
count = 0
shapePredioctor = './data/dlib/shape_predictor_68_face_landmarks.dat'

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredioctor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
res = ""
while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if count > incontinuity:
        rects = detector(gray, 1)
        # print(len(rects))
        # a睡觉  b不睡觉
        a, b = 0, 0
        for rect in rects:
            # 获取人脸方框坐标
            pos_start = tuple([rect.left() - 5, rect.top() - 5])
            pos_end = tuple([rect.right() + 5, rect.bottom() + 5])

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eyeAspectRatio(leftEye)
            rightEAR = eyeAspectRatio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            # 闭眼了
            if ear < earThresh:
                cv2.rectangle(frame, pos_start, pos_end, (0, 255, 0), 2)
                a += 1
                cv2.rectangle(frame, pos_start, pos_end, (0, 0, 255), 2)
                break
            else:  # 没闭眼
                b += 1
                # 画绿方框
                cv2.rectangle(frame, pos_start, pos_end, (0, 255, 0), 2)

        print("睡觉:", a, "不睡觉:", b)
        if a + b:
            c = a / (a + b)
            count = 0
            res = "Sleep:" + str(c) + "   Rise rate:" + str(1 - c)
        # print(res)
        # 显示睡觉率,抬头率



    count += 1
    cv2.putText(frame, res, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('class', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print('quit')
        break

cam.release()
cv2.destroyAllWindows()
