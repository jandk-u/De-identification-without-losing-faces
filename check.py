import cv2
import numpy as np

from utils import *
from model import *

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T


def check(image_path):
    model = ModelTrainning()
    model.eval()
    check_point = torch.load('weight/epoch_19.pth')
    model.load_state_dict(state_dict=check_point['model_state_dict'])

    transform_val = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    transform_image = T.ToPILImage()
    # make point origin image
    face_ori = crop_face(image_path)
    cv2.imwrite('sample_out_1.png', cv2.cvtColor(face_ori, cv2.COLOR_BGR2RGB))
    # landmarkpoints = get_landmark(face_ori)
    # print(len(landmarkpoints[0]))
    # new_point = make_point(landmark_point=landmarkpoints[0])

    # make point gener image
    face_ori = transform_val(image=face_ori)['image']
    face_ori = face_ori.unsqueeze(0)
    face_gen = model.forward_one(face_ori)
    face_gen = face_gen.squeeze(0)
    face_gen = transform_image(face_gen)
    face_gen = cv2.cvtColor(np.array(face_gen), cv2.COLOR_BGR2RGB)
    cv2.imwrite('sample_out_2.png', cv2.cvtColor(face_gen, cv2.COLOR_RGB2BGR))
    landmarkpoints_gen = get_landmark(face_gen)
    print(landmarkpoints_gen[0])
    new_point_gen = make_point(landmark_point=landmarkpoints_gen[0])

    rect = cv2.boundingRect(np.array(new_point_gen))
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(new_point_gen)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    # print(triangles)
    # print(landmarkpoints_gen)


    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = transform_val(image=image)['image']
    landmarkpoints = get_landmark(image)
    # print(len(landmarkpoints[0]))
    new_point = make_point(landmark_point=landmarkpoints[0])
    rect_org = cv2.boundingRect(np.array(new_point))
    subdiv_org = cv2.Subdiv2D(rect_org)
    subdiv_org.insert(new_point)
    triangles_org = subdiv_org.getTriangleList()
    triangles_org = np.array(triangles_org, dtype=np.int32)
    # print(image)
    # print(type(np.asarray(new_point, dtype=np.float32)))

    for point, point1 in zip(triangles, triangles_org):
        warp_mat = cv2.getAffineTransform(np.asarray(point, dtype=np.float32), np.asarray(point1, dtype=np.float32))

        warp_dst = cv2.wrapAffine(image, warp_mat, (image.shape[1], image.shape[0]))

    cv2.imshow('Source image', image)
    cv2.imshow('Warp', warp_dst)

    cv2.waitKey(0)


if __name__ == '__main__':
    check('1.jpg')