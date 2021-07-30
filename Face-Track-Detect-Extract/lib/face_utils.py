import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils


def judge_side_face(facial_landmarks):
    wide_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[1])
    high_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[3])
    dist_rate = high_dist / wide_dist

    # cal std
    vec_A = facial_landmarks[0] - facial_landmarks[2]
    vec_B = facial_landmarks[1] - facial_landmarks[2]
    vec_C = facial_landmarks[3] - facial_landmarks[2]
    vec_D = facial_landmarks[4] - facial_landmarks[2]
    dist_A = np.linalg.norm(vec_A)
    dist_B = np.linalg.norm(vec_B)
    dist_C = np.linalg.norm(vec_C)
    dist_D = np.linalg.norm(vec_D)

    # cal rate
    high_rate = dist_A / dist_C
    width_rate = dist_C / dist_D
    high_ratio_variance = np.fabs(high_rate - 1.1)  # smaller is better
    width_ratio_variance = np.fabs(width_rate - 1)

    return dist_rate, high_ratio_variance, width_ratio_variance

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	EAR = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return EAR

def eye_open(gray, bbox, shape_predictor, EYE_THRESH):
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    left, top, right, bottom = bbox
    shape = shape_predictor(gray, dlib.rectangle(left, top, right, bottom))
    shape = face_utils.shape_to_np(shape)
    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    # print(leftEAR, rightEAR)
    if leftEAR > EYE_THRESH and rightEAR > EYE_THRESH: return True
    return False