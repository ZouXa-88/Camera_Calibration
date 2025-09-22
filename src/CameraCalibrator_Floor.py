import cv2
import numpy as np
import pandas as pd
import yaml
from easydict import EasyDict
import os
import matplotlib.pyplot as plt
import pickle

from parameter2position import parameter2position, show_cameras

def getConfig(configPath: str):
    config = None
    with open(configPath, 'rb') as f:
        config = EasyDict(yaml.load(f, yaml.FullLoader))

    return config

def getGridPoints(excelPath: str):
    data = pd.read_excel(excelPath, sheet_name=None, header=None)
    for camera, points in data.items():
        data[camera] = points.values.tolist()

    return data

def getIntrinsic(pklFile: str):
    if os.path.exists(pklFile):
        with open(pklFile, 'rb') as f:
            return pickle.load(f)
    else:
        print(f'No intrinsic pickle file found: {pklFile}')
        exit(-1)
    

def packPoints(CHESSBOARD_SIZE, SQUARE_SIZE, gridPoints, CENTER):
    worldPoints = []
    imagePoints = []

    for i in range(CHESSBOARD_SIZE[0]):
        for j in range(CHESSBOARD_SIZE[1]):
            try:
                if type(gridPoints[i][j]) is not float:
                    imagePoints.append(list(eval(gridPoints[i][j])))
                    worldPoints.append([(j - CENTER[1]) * SQUARE_SIZE, (CENTER[0] - i) * SQUARE_SIZE, 0.0])
            except IndexError:
                continue

    return np.array(worldPoints, dtype=np.float32), np.array(imagePoints, dtype=np.float32)

def getK(worldPoints, imagePoints, imageShape):
    '''
    =============
    Don't use it
    =============
    
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
    ret, mtx, dist, rvec, tvec = cv2.calibrateCamera([worldPoints], [imagePoints], imageShape, None, None) # 3-dimension array
    
    print(f'K:\n{mtx}')
    print(f'Distortion:\n{dist}')
    print(f'Reprojection Error: {ret}')
    
    return mtx, dist

def getRT(worldPoints, imagePoints, K, dist=np.zeros(4)):
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
    success, rvec, tvec = cv2.solvePnP(worldPoints, imagePoints, K, dist)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Print the extrinsic parameters
    print("Rotation Matrix (R):\n", R)
    print("Translation Vector (t):\n", tvec)

    # Reprojection error
    # --------------------
    # Step 1: Project 3D points back to 2D using R and t
    reprojected_points, _ = cv2.projectPoints(worldPoints, rvec, tvec, K, dist)

    # Convert to (N,2) shape
    reprojected_points = reprojected_points.reshape(-1, 2)

    # Step 2: Compute Euclidean distance between actual and reprojected points
    errors = np.linalg.norm(imagePoints - reprojected_points, axis=1)

    # Step 3: Compute the average reprojection error
    mean_error = np.mean(errors)
    print(f'Reprojection Error: {mean_error:.4f}')
    # --------------------

    return R, tvec

if __name__ == '__main__':
    DATA_ROOT = '/home/user/Desktop/20250917/calibration'
    SAVE_RESULT = False
    SHOW_CAMERAS = True
    config = getConfig(f'{DATA_ROOT}/extrinsic/config.yaml')

    # Fetch configurations
    EXCEL_FILE = config.EXCEL_FILE
    CHESSBOARD_SIZE = (config.CHESSBOARD.SHAPE.ROWS, config.CHESSBOARD.SHAPE.COLS)
    SQUARE_SIZE = config.CHESSBOARD.SQUARE_SIZE
    CENTER = (config.CHESSBOARD.CENTER.ROW, config.CHESSBOARD.CENTER.COL)
    CAMERA_IDS = config.CAMERA_IDS
    RESOLUTION = config.RESOLUTION
    SHARED_INTRINSIC = config.SHARED_INTRINSIC
    
    try:
        gridPoints = getGridPoints(f'{DATA_ROOT}/extrinsic/{EXCEL_FILE}')
    except FileNotFoundError:
        print(f'Can\'t find the excel file: {EXCEL_FILE}')
        exit(1)
    
    # Fetch intrinsic matrix
    # Search intrinsic matrices of all cameras first
    intrinsic = {}
    for camera_dir in os.listdir(f'{DATA_ROOT}/intrinsic'):
        camera_intrinsic_file = f'{DATA_ROOT}/intrinsic/{camera_dir}/camera_parameter.pkl'
        if os.path.exists(camera_intrinsic_file):
            intrinsic[camera_dir] = getIntrinsic(camera_intrinsic_file)
    # Replace it with shared intrinsic matrix if possible
    if SHARED_INTRINSIC.USE_SHARED_INTRINSIC:
        try:
            shared_intrinsic = intrinsic[f'camera{SHARED_INTRINSIC.CAMERA_ID}']
            for camera_name in intrinsic.keys():
                intrinsic[camera_name] = shared_intrinsic
        except KeyError:
            print(f'Error: You set \'USE_SHARED_INTRINSIC\' as true, but \'camera{SHARED_INTRINSIC.CAMERA_ID}\' is not found')
            exit(1)
    
    '''
    K = np.array([
        [567.16, 0, 319.24],
        [0, 568.39, 282.18],
        [0, 0, 1],
    ])
    distortion = np.array([[-0.22457275, 0.60590569, 0.0082039, 0.01072181, -2.17135628]])
    '''

    Ks, RTs, Ps = [], [], []
    
    for camera_id in CAMERA_IDS:
        camera_name = f'camera{camera_id}'
        worldPoints, imagePoints = packPoints(CHESSBOARD_SIZE, SQUARE_SIZE, gridPoints[camera_name], CENTER)
        
        #K, dist = getK(worldPoints, imagePoints, RESOLUTION)
        
        #R, T = getRT(worldPoints, imagePoints, K)
        R, T = getRT(worldPoints, imagePoints, intrinsic[camera_name]['K'])
        #R, T = getRT(worldPoints, imagePoints, intrinsic['K'])

        #Ks.append(K)
        Ks.append(intrinsic[camera_name]['K'])
        #Ks.append(intrinsic['K'])
        RTs.append(np.hstack((R, T)))
        Ps.append(Ks[-1] @ RTs[-1])
        
    savedData = {'K': np.array(Ks), 'RT': np.array(RTs), 'P': np.array(Ps)}
    print(savedData)

    if SAVE_RESULT:
        with open(f'{DATA_ROOT}/camera_parameter.pickle', 'wb') as f:
            pickle.dump(savedData, f)
            
    if SHOW_CAMERAS:
        show_cameras(parameter2position(savedData))
