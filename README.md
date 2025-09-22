# Camera_Calibration
Calibrate the camera, outputting intrinsic and extrinsic matrix

## Data Preparation
About the format of config.yaml and points.xlsx, you can refer to /example.\
The directory tree of prepared calibration data should look like below:
```
{ROOT}
    |-- intrinsic
        |-- camera0
            |-- checkerboard001.jpg
            |-- checkerboard002.jpg
            |-- checkerboard003.jpg
            ...
            |-- config.yaml
            |-- camera_parameter.pkl <-- The output of CameraCalibrator_Checkerboard.py
        |-- camera1
        ...
    |-- extrinsic
        |-- config.yaml
        |-- points.xlsx
    |-- camera_parameter.pickle <-- The output of CameraCalibrator_Floor.py
```
