# Camera Calibration using Charuco Board

This repository contains the code to calibrate a camera using a *Charuco board*. The code uses the OpenCV library to detect the Charuco board and the chessboard pattern. The calibration is done using the OpenCV functions `calibrateCamera()`. The repository also includes a sample image of a Charuco board and the chessboard pattern.

> The code is based on my [aruco wraper project](https://github.com/What-Pity/aruco), if you want to gennerate a Charuco board, you can use the `demo_generate.py` in that project

## Requirements

- Python 3.6+
- OpenCV 4.8

## Usage

Call `calibration.py` directly with the path to the calibration images, image suffix, and output file name. You can open `demo.cmd` with a text editor and modify it according to your environment to avoid typing the path and suffix every time.

***More details are provided in the comments of the code in Chinese.***

## Waring

`calibration.py` is written for my aruco wrapper project and use the default pattern whose dictionaryID is `cv2.aruco.DICT_6X6_50`. If you want to use other patterns, you need to modify the `calibration.py` accordingly (`bd = charucoBoard()` in line 34), otherwise it may not work correctly.

