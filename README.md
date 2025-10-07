# Camera_Calibration
Calibrate the camera, outputting intrinsic and extrinsic matrix

## Data Preparation
For the format of config.yaml and points.xlsx, you can refer to [example](https://github.com/ZouXa-88/Camera_Calibration/tree/main/example).\
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

## How to run
### Intrinsic
```
python CameraCalibrator_Checkerboard.py \
    {your_path_to_root} \
    --camera_name {camera_name} \
    --run_only
```
* ```--camera_name```: The folder name of the camera you want to calibrate, under intrinsic folder.
* ```--run_only```: Optional. Set this flag if you don't want to save the result, since the program will replace the previous file by default.
### Extrinsic
```
python CameraCalibrator_Floor.py \
    {your_path_to_root} \
    --run_only
```
* ```--run_only```: Optional. Set this flag if you don't want to save the result, since the program will replace the previous file by default.
