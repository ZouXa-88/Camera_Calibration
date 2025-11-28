import cv2
import numpy as np
import yaml
from easydict import EasyDict
import os
import random
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('root', help='The path to the root of calibration data')
    parser.add_argument('--camera_name', required=True, help='The folder name of the camera you want to calibrate')
    parser.add_argument('--run_only', action='store_true', help='Don\'t save the result, since it will replace previous file')
    
    args = parser.parse_args()
    return args

def get_config(config_path: str):
    try:
        with open(config_path, 'rb') as f:
            return EasyDict(yaml.load(f, yaml.FullLoader))
    except Exception as e:
        print(e)
        exit(1)

def get_images(folder: str, extensions: list[str]):
    if os.path.exists(folder):
        images = [f'{folder}/{image}' for image in os.listdir(folder) if ('.' + image.split('.')[-1]) in extensions]
    else:
        print(f'Error: No such folder: {folder}')
        exit(1)
        
    print(f'Total number of images: {len(images)}')
    return images

# Intrinsic matrix
def calibrate_camera(images: list[str], checkerboard_size: iter, square_size: int, resolution: iter):
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    count = 0
    max_num_images = min(50, len(images)) # It will take a long time to compute if the number of images is large
    
    random.shuffle(images) # Randomly pick images
    
    with tqdm(total=len(images), desc='Detecting checkerboard from images', unit='data', ncols=100) as pbar:
        for img_path in images:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None, flags=flags)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                count += 1

            pbar.update(1)
            pbar.set_postfix_str(f"Collected: {count}/{max_num_images}")

            if count >= max_num_images:
                pbar.write(f"✅ Target of {max_num_images} samples reached! Stopping iteration.")
                break
            
    print(f'Number of images to compute intrinsic matrix: {count}')
    print(f'Computing intrinsic matrix...')
    ret, mtx, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, resolution, None, None)
    
    return ret, mtx, dist, rvec, tvec

if __name__ == '__main__':
    # Fetch arguments
    args = parse_args()
    ROOT = args.root
    CAMERA_NAME = args.camera_name
    SAVE_RESULT = not args.run_only
    
    # Fetch configurations
    config = get_config(f'{ROOT}/intrinsic/{CAMERA_NAME}/config.yaml')
    CHESSBOARD_SIZE = (config.CHESSBOARD.SHAPE.COLS, config.CHESSBOARD.SHAPE.ROWS) # (x, y)
    RESOLUTION = config.RESOLUTION
    
    images = get_images(f'{ROOT}/intrinsic/{CAMERA_NAME}', config.FILE_EXTENSIONS)

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
        saved_data = {
            'K': mtx,
            'Dist': dist,
        }
        
        with open(f'{ROOT}/intrinsic/{CAMERA_NAME}/camera_parameter.pkl', 'wb') as f:
            pickle.dump(saved_data, f)
