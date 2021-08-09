import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from math import hypot
import dlib

emotion = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_count = [0, 0, 0, 0, 0, 0, 0]
left_eye_idx = [36, 37, 38, 39, 40, 41]
right_eye_idx = [42, 43, 44, 45, 46, 47]
output = np.empty((2, 0))
# frame_data = np.empty((9, 0))

prev_landmarks = [0]

model = keras.models.load_model("models/_mini_XCEPTION.106-0.65.hdf5")
face_cas = cv2.CascadeClassifier(
    'C:\\Users\\Admin\\anaconda3\\envs\\ex2\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_gaze_ratio(frame, gray, eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    if min_y == max_y: max_y += 1
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    if ver_line_lenght != 0:
        ratio = hor_line_lenght / ver_line_lenght
    else:
        ratio = 0
    return ratio


def get_frame_features(frame, is_first_frame):
    """
    general feature getter.
    :param frame - input frame
    :return: an np array with 9 columns, containing the expression rate, gaze direction, and number of faces (o if none)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    num_faces = 0 if len(faces) == 0 else 1

    # if there's no face in the frame, return array of zeros
    if num_faces == 0:
        return

    # Facial expression
    face = faces[0]  # I only care about one face, arbitrarily choosing the first one detected
    x, y = face.left(), face.top()
    w, h = face.right() - x, face.bottom() - y
    face_component = gray[y:y + h, x:x + w]
    # print(face_component.shape)
    # print(face_component.shape[0])
    # print(face_component.shape[1])
    if face_component.shape[0] != 0 and face_component.shape[1] != 0:
        fc = cv2.resize(face_component, (48, 48))
        inp = np.reshape(fc, (1, 48, 48, 1)).astype(np.float32)
        inp = inp / 255.
        expression_prediction = model.predict(inp)

        emotion_count[np.argmax(expression_prediction)] += 1
        # print(expression_prediction[0])
        # print(np.argmax(expression_prediction[0]))


def get_movement_between_frames(landmarks_f1, landmarks_f2):
    xy_1 = np.array([[p.x, p.y] for p in landmarks_f1.parts()])
    xy_2 = np.array([[p.x, p.y] for p in landmarks_f2.parts()])

    # diff = (np.abs(xy_2 - xy_1)).sum()
    diff = (xy_2 - xy_1).sum()
    return diff


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_emotion_count():
    return emotion_count
