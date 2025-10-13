import cv2
import os
from pathlib import Path
import argparse

def video_to_images(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_name = f"{frame_count:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, img_name), frame)
        frame_count += 1

    cap.release()
    print(f"完成 {video_path} -> {output_dir} ({frame_count} 張圖片)")
    
def get_path():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('path', help='The path to the folder of videos')
    
    args = parser.parse_args()
    return args.path

if __name__ == "__main__":
    recording_dir = Path(get_path())

    for video_file in recording_dir.glob("*.mkv"):  # 假設影片是 mp4
        # 取檔名中的 cameraX
        camera_name = video_file.stem.split("_")[-1]
        output_dir = recording_dir / camera_name  # 存到 recording_dir 下的相機資料夾
        video_to_images(video_file, output_dir)
