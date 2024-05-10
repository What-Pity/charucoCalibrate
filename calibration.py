"""使用charucoBoard估计相机内参

参数：
impath: 图像文件夹路径，默认./data
suffix: 图像文件后缀，默认jpg
output: 输出文件夹路径（camera_matrix.csv存放相机内参，distortion_coefficients.csv存放畸变系数），默认./camera_params

使用说明：python calibration.py --impath ./images --suffix png --output ./output --szie 50
简单用法：python charuco_calibration.py

"""
import cv2
from aruco import charucoBoard
import numpy as np
from pathlib import Path
import argparse

# 初始化
parser = argparse.ArgumentParser(description='charuco calibration')
parser.add_argument('--impath', type=str,
                    default="./data", help='directory of images')
parser.add_argument('--suffix', type=str, default="jpg",
                    help='suffix of images')
parser.add_argument('--output', type=str, default="./camera_params",
                    help='output directory of camera parameters')
parser.add_argument('--size', type=int, default=57,
                    help='size of charuco board grid in mm')
args = parser.parse_args()

directory = Path(args.impath)
suffix = args.suffix
save_dir = Path(args.output)

bd = charucoBoard()

corners = []
ids = []
objPoints = []

# 世界坐标
sz = np.array(bd.board_size)-1
col = sz[0]
row = sz[1]
objp = np.zeros((col*row, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)
grid_size = args.size  # 格子大小
objp *= grid_size  # 单位换算

num_file = 0
for img_name in directory.glob("*."+suffix):
    img = cv2.imread(str(img_name))
    charucoCorners, charucoIds, markerCorners, markerIds = bd.detect(img)
    corners.append(charucoCorners)
    ids.append(charucoIds)
    objPoints.append(objp)
    num_file += 1
print(f"Detect {num_file} images.")


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objPoints, corners, img[:, :, 0].shape[::-1], None, None)
print(f"reprojection error: {ret}")
save_dir.mkdir(exist_ok=True)
np.savetxt(save_dir/"camera_matrix.csv", mtx, delimiter=",")
np.savetxt(save_dir/"distortion_coefficients.csv", dist, delimiter=",")
