import cv2
import numpy as np
import yaml
from easydict import EasyDict
import os
import random
import matplotlib.pyplot as plt
import pickle

def get_config(config_path: str):
    config = None
    with open(config_path, 'rb') as f:
        config = EasyDict(yaml.load(f, yaml.FullLoader))

    return config

def get_images(folder: str, extensions: list[str]):
    if os.path.exists(folder):
        images = [f'{folder}/{image}' for image in os.listdir(folder) if ('.' + image.split('.')[-1]) in extensions]
    else:
        print(f'Error: No such folder: {folder}')
        exit(-1)
        
    print(f'Total number of images: {len(images)}')
    return images

# Intrinsic parameters
def calibrate_camera(images: list[str], checkerboard_size: iter, square_size: int, resolution: iter):
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    count = 0
    max_num_images = min(50, len(images)) # It will take a long time to compute if the number of images exceeds 50
    
    random.shuffle(images) # Randomly pick images
    
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None, flags=flags)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            count += 1
        if count >= max_num_images:
            break
            
    print(f'Number of images to compute intrinsic matrix: {count}')
    ret, mtx, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, resolution, None, None)
    
    return ret, mtx, dist, rvec, tvec

if __name__ == '__main__':
    DATA_ROOT = '/home/user/Desktop/20250917/calibration'
    CAMERA_NAME = 'camera3'
    SAVE_RESULT = False
    
    config = get_config(f'{DATA_ROOT}/intrinsic/{CAMERA_NAME}/config.yaml')

    CHESSBOARD_SIZE = (config.CHESSBOARD.SHAPE.COLS, config.CHESSBOARD.SHAPE.ROWS) # (x, y)
    RESOLUTION = config.RESOLUTION
    images = get_images(f'{DATA_ROOT}/intrinsic/{CAMERA_NAME}', config.FILE_EXTENSIONS)

    print(f'Computing...')
    ret, mtx, dist, rvec, tvec = calibrate_camera(
        images, 
        CHESSBOARD_SIZE, 
        config.CHESSBOARD.SQUARE_SIZE,
        RESOLUTION,
    )

    print(f'K:\n{mtx}')
    print(f'Distortion:\n{dist}')
    print(f'Reprojection Error: {ret}')

    if SAVE_RESULT:
        savedData = {
            'K': mtx,
            'Dist': dist,
        }
        
        with open(f'{DATA_ROOT}/intrinsic/{CAMERA_NAME}/camera_parameter.pkl', 'wb') as f:
            pickle.dump(savedData, f)
