import os

import dlib
import mediapipe as mp
import cv2
from tqdm import tqdm
import torch


def create_root_data(root_dir, save_dir):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for idx, file in tqdm(enumerate(os.listdir(root_dir))):
        try:
            path = os.path.join(root_dir, file)
            image = cv2.imread(path)

            h, w, c = image.shape
            rs = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not rs.detections:
                continue
            # print(rs.detections)
            bbox = rs.detections[0].location_data.relative_bounding_box
            x = round(bbox.xmin*w)
            y = round(bbox.ymin*h)
            w = round(bbox.width*w)
            h = round(bbox.height*h)
            crop_image = image[y: y+h, x: x+w]
            crop_image = cv2.resize(crop_image, (128, 128))
            save_path = os.path.join(save_dir, str(idx) + '.jpg')
            cv2.imwrite(save_path, crop_image)
        except Exception as e:
            print(e)

    print("Create face data complete!")


def crop_face(image):
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        image = cv2.imread(image)

        h, w, c = image.shape
        rs = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # print(rs.detections)
        bbox = rs.detections[0].location_data.relative_bounding_box
        x = round(bbox.xmin * w)
        y = round(bbox.ymin * h)
        w = round(bbox.width * w)
        h = round(bbox.height * h)
        crop_image = image[y: y + h, x: x + w]
        crop_image = cv2.resize(crop_image, (64, 64))
    except Exception as e:
        raise e

    return crop_image


def save_weight(model, epoch, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_weight(path):
    check_point = torch.load(path)
    return check_point


def make_point(landmark_point):
    print("--------")
    print(landmark_point)
    print(len(landmark_point))
    point_18 = landmark_point[17]
    point_26 = landmark_point[25]
    point_49 = landmark_point[48]
    point_54 = landmark_point[53]

    new_point_left = []
    new_point_right = []
    for i in range(1, 5, 1):
        if i == 1:
            x_new_point_left = round(point_18[0] + (i/15)*(point_49[0] - point_18[0]))
            y_new_point_left = round(point_18[1] + (i/5)*(point_49[1] - point_18[1]))

            x_new_point_right = round(point_26[0] + (i / 15)*(point_54[0] - point_26[0]))
            y_new_point_right = round(point_26[1] + (i / 5)*(point_54[1] - point_26[1]))

            new_point_left.append((x_new_point_left, y_new_point_left))
            new_point_right.append((x_new_point_right, y_new_point_right))
        else:
            x_new_point_left = round(new_point_left[i-2][0] + (i / 15)*(point_49[0] - point_18[0]))
            y_new_point_left = round(point_18[1] + (i / 5)*(point_49[1] - point_18[1]))

            x_new_point_right = round(new_point_right[i-2][0] + (i / 15)*(point_54[0] - point_26[0]))
            y_new_point_right = round(point_26[1] + (i / 5)*(point_54[1] - point_26[1]))

            new_point_left.append((x_new_point_left, y_new_point_left))
            new_point_right.append((x_new_point_right, y_new_point_right))
    results = landmark_point + new_point_left + new_point_right
    return results


def get_landmark(img, predictor_path='shape_predictor_68_face_landmarks.dat'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    det = detector(img, 1)
    faces = []
    for k, d in enumerate(det):
        landmarks = predictor(img, d)
        landmarks_rs = []
        for i in range(68):
            landmarks_rs.append((landmarks.part(i).x, landmarks.part(i).y))
        faces.append(landmarks_rs)
    return faces


if __name__ == '__main__':
    create_root_data('/home/j/Downloads/05000/', '/home/j/Learn/AI/FATM/save_dir')
    # make_point('')
    # landmarkpoints = get_landmark(image='000211.jpg', predictor_path='shape_predictor_68_face_landmarks.dat')
    # new_point = make_point(landmark_point=landmarkpoints[0])
    # print(new_point)
