from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np




def eyeAspectRatio(eye):
	#vertical
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	#horizontal
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

count = 0

earThresh = 0.20 # 闭眼阈值
earFrames = 20 # 闭眼连续40帧
shapePredioctor = './data/dlib/shape_predictor_68_face_landmarks.dat'

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredioctor)


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

while True:
	_, frame = cam.read()
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 1)
	# print(len(rects))


	for rect in rects:
		# 获取人脸方框坐标
		pos_start = tuple([rect.left()-5, rect.top()-5])
		pos_end = tuple([rect.right()+5, rect.bottom()+5])


		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eyeAspectRatio(leftEye)
		rightEAR = eyeAspectRatio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		# print(ear)
		# 闭眼了
		if ear < earThresh:
			count = count + 1
			cv2.rectangle(frame, pos_start, pos_end, (0, 255, 0), 2)


			if count >= earFrames:

				cv2.rectangle(frame, pos_start, pos_end, (0, 0, 255), 2)
				cv2.putText(frame, "SLEEP!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

				break

		else:

			count = 0
			# 画眼睛
			# cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			# cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			# 画绿方框
			cv2.rectangle(frame, pos_start, pos_end, (0, 255, 0), 2)




	cv2.imshow('class', frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord('q'):
		print('quit')
		break

cam.release()
cv2.destroyAllWindows()