import cv2
import numpy as np
import pandas as pd
import yaml
from easydict import EasyDict
import os
import matplotlib.pyplot as plt
import pickle
import argparse

from parameter2position import parameter2position, show_cameras

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('root', help='The path to the root of calibration data')
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

def get_grid_points(excel_path: str):
    try:
        data = pd.read_excel(excel_path, sheet_name=None, header=None)
        for camera, points in data.items():
            data[camera] = points.values.tolist()

        return data
    except Exception as e:
        print(e)
        exit(1)

def get_intrinsic(pkl_file: str):
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)
    else:
        print(f'No intrinsic pickle file found: {pkl_file}')
        exit(1)
    
def get_chessboard(CHESSBOARDS: list, camera_id: int):
    for CHESSBOARD in CHESSBOARDS:
        if camera_id in CHESSBOARD.CAMERA_IDS:
            return CHESSBOARD
        
    print(f'Error: Camera id \'{camera_id}\' isn\'t in any CHESSBOARDS')
    exit(1)

def pack_points(CHESSBOARD, grid_points):
    world_points = []
    image_points = []

    for i in range(CHESSBOARD.SHAPE.ROWS):
        for j in range(CHESSBOARD.SHAPE.COLS):
            try:
                if type(grid_points[i][j]) is not float:
                    image_points.append(list(eval(grid_points[i][j])))
                    world_points.append([
                        (j - CHESSBOARD.CENTER.EXCEL_INDEX.COL) * CHESSBOARD.SQUARE_SIZE + CHESSBOARD.CENTER.COORDINATE_3D.X, 
                        (CHESSBOARD.CENTER.EXCEL_INDEX.ROW - i) * CHESSBOARD.SQUARE_SIZE + CHESSBOARD.CENTER.COORDINATE_3D.Y, 
                        0.0
                    ])
            except IndexError:
                continue

    return np.array(world_points, dtype=np.float32), np.array(image_points, dtype=np.float32)

def get_RT(world_points, image_points, K, dist=np.zeros(4)):
    '''
    world_points = np.array([
        [X1, Y1, 0],  # Point 1
        [X2, Y2, 0],  # Point 2
        [X3, Y3, 0],  # Point 3
        [X4, Y4, 0]   # Point 4
    ], dtype=np.float32)

    image_points = np.array([
        [u1, v1],  # Point 1
        [u2, v2],  # Point 2
        [u3, v3],  # Point 3
        [u4, v4]   # Point 4
    ], dtype=np.float32)
    '''

    # Solve for rotation and translation
    success, rvec, tvec = cv2.solvePnP(world_points, image_points, K, dist)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Print the extrinsic parameters
    print("Rotation Matrix (R):\n", R)
    print("Translation Vector (t):\n", tvec)

    # Reprojection error
    # --------------------
    # Step 1: Project 3D points back to 2D using R and t
    reprojected_points, _ = cv2.projectPoints(world_points, rvec, tvec, K, dist)

    # Convert to (N,2) shape
    reprojected_points = reprojected_points.reshape(-1, 2)

    # Step 2: Compute Euclidean distance between actual and reprojected points
    errors = np.linalg.norm(image_points - reprojected_points, axis=1)

    # Step 3: Compute the average reprojection error
    mean_error = np.mean(errors)
    print(f'Reprojection Error: {mean_error:.4f}')
    # --------------------

    return R, tvec

if __name__ == '__main__':
    # Fetch arguments
    args = parse_args()
    ROOT = args.root
    SAVE_RESULT = not args.run_only

    # Fetch configurations
    config = get_config(f'{ROOT}/extrinsic/config.yaml')
    EXCEL_FILE = config.EXCEL_FILE
    CHESSBOARDS = config.CHESSBOARDS
    ALL_CAMERA_IDS = config.ALL_CAMERA_IDS
    SHARED_INTRINSIC = config.SHARED_INTRINSIC
    
    # Get labeled points from excel file
    grid_points = get_grid_points(f'{ROOT}/extrinsic/{EXCEL_FILE}')
    
    # Get intrinsic matrix
    # Search intrinsic matrices of all cameras first
    intrinsic = {}
    for camera_dir in os.listdir(f'{ROOT}/intrinsic'):
        camera_intrinsic_file = f'{ROOT}/intrinsic/{camera_dir}/camera_parameter.pkl'
        if os.path.exists(camera_intrinsic_file):
            intrinsic[camera_dir] = get_intrinsic(camera_intrinsic_file)
    # Replace it with shared intrinsic matrix if possible
    if SHARED_INTRINSIC.USE_SHARED_INTRINSIC:
        try:
            shared_intrinsic = intrinsic[f'camera{SHARED_INTRINSIC.CAMERA_ID}']
            for camera_name in intrinsic.keys():
                intrinsic[camera_name] = shared_intrinsic
        except KeyError:
            print(f'Error: You set \'USE_SHARED_INTRINSIC\' as true, but \'camera{SHARED_INTRINSIC.CAMERA_ID}\' is not found')
            exit(1)

    Ks, RTs, Ps = [], [], []
    
    for camera_id in ALL_CAMERA_IDS:
        CHESSBOARD = get_chessboard(CHESSBOARDS, camera_id)
        camera_name = f'camera{camera_id}'
        world_points, image_points = pack_points(CHESSBOARD, grid_points[camera_name])
        print(world_points)
        
        R, T = get_RT(world_points, image_points, intrinsic[camera_name]['K'])

        Ks.append(intrinsic[camera_name]['K'])
        RTs.append(np.hstack((R, T)))
        Ps.append(Ks[-1] @ RTs[-1])
        
    saved_data = {'K': np.array(Ks), 'RT': np.array(RTs), 'P': np.array(Ps)}
    print(saved_data)

    if SAVE_RESULT:
        with open(f'{ROOT}/camera_parameter.pickle', 'wb') as f:
            pickle.dump(saved_data, f)
            
    # Show cameras
    show_cameras(parameter2position(saved_data))
