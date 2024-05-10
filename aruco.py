import cv2
import numpy as np


class aruco:
    def __init__(self, dictionary_id=cv2.aruco.DICT_6X6_50):
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary, self.parameters)

    def generate(self, id, size):
        # 生成aruco标记
        # 生成标记
        markerImage = np.zeros((size, size), dtype=np.uint8)
        markerImage = cv2.aruco.generateImageMarker(
            self.dictionary, id, size)
        return markerImage

    def detect(self, image):
        # 检测aruco标记
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(image)
        return corners, ids, rejectedImgPoints

    def draw(self, image, corners, ids, pose=None, axis_size=1):
        # 绘制aruco标记
        image_copy = image.copy()
        image_copy = cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)
        if pose:
            for i in range(len(corners)):
                image_copy = cv2.drawFrameAxes(
                    image_copy, pose["cameraMatrix"], pose["distCoeffs"], pose["rvec"][i], pose["tvec"][i], axis_size)
        return image_copy

    def pose_estimate(self, corners, cameraMatrix, distCoeffs):
        # 估计aruco标记姿态
        markerNum = len(corners)
        objPoints = np.array(
            [[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]])
        pose = {"cameraMatrix": cameraMatrix,
                "distCoeffs": distCoeffs,
                "rvec": np.zeros((markerNum, 3)),
                "tvec": np.zeros((markerNum, 3))}
        for i, corner in enumerate(corners):
            retval, rvec, tvec = cv2.solvePnP(
                objPoints, corner, cameraMatrix, distCoeffs)
            if retval:
                pose["rvec"][i] = np.squeeze(rvec)
                pose["tvec"][i] = np.squeeze(tvec)
            else:
                Warning("PnP solve failed, Pose return None.")
                return None
        return pose


class arucoBoard(aruco):
    def __init__(self, dictionary_id=cv2.aruco.DICT_6X6_50, board_size=(5, 7), marker_square_rate=0.1):
        super().__init__(dictionary_id)
        self.board_size = board_size
        self.board = cv2.aruco.GridBoard(
            board_size, 1, marker_square_rate, self.dictionary)

    def generate(self, size=(500, 700)):
        boardImage = self.board.generateImage(size, marginSize=10)
        return boardImage

    def detect(self, image, refine=True):
        corners, ids, rejectedImgPoints = super().detect(image)
        if refine:
            corners, ids, rejectedImgPoints, _ = self.detector.refineDetectedMarkers(
                image, self.board, corners, ids, rejectedImgPoints)
        return corners, ids, rejectedImgPoints

    def pose_estimate(self, corners, ids, cameraMatrix, distCoeffs):
        obj_points, img_points = self.board.matchImagePoints(corners, ids)
        retval, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, cameraMatrix, distCoeffs)
        if retval:
            pose = {"cameraMatrix": cameraMatrix,
                    "distCoeffs": distCoeffs,
                    "rvec": rvec,
                    "tvec": tvec}
            return pose
        else:
            Warning("PnP solve failed, Pose return None.")
            return None

    def draw(self, image, corners, ids, pose=None, axis_size=1):
        img = super().draw(image, corners, ids, axis_size=axis_size)
        if pose:
            img = cv2.drawFrameAxes(
                img, pose["cameraMatrix"], pose["distCoeffs"], pose["rvec"], pose["tvec"], axis_size)
        return img


class charucoBoard(arucoBoard):
    def __init__(self, dictionary_id=cv2.aruco.DICT_6X6_50, board_size=(5, 7), marker_square_rate=0.6, ids=None):
        if marker_square_rate >= 1:
            marker_square_rate = 1
        super().__init__(dictionary_id, board_size, marker_square_rate)
        self.board = cv2.aruco.CharucoBoard(
            self.board_size, 1, marker_square_rate, self.dictionary, ids)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)

    def detect(self, image, refine=False):
        charucoCorners, charucoIds, markerCorners, markerIds = self.charuco_detector.detectBoard(
            image)
        return charucoCorners, charucoIds, markerCorners, markerIds

    def draw(self, img, charucoCorners, charucoIds, pose=None, charuco_color=(0, 0, 255), axis_size=1):
        img = cv2.aruco.drawDetectedCornersCharuco(
            img, charucoCorners, charucoIds, charuco_color)
        if pose:
            img = cv2.drawFrameAxes(
                img, pose["cameraMatrix"], pose["distCoeffs"], pose["rvec"], pose["tvec"], axis_size)
        return img

    def pose_estimate(self, charucoCorners, charucoIds, cameraMatrix, distCoeffs):
        obj_points, img_points = self.board.matchImagePoints(
            charucoCorners, charucoIds)
        retval, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, cameraMatrix, distCoeffs)
        if retval:
            pose = {"cameraMatrix": cameraMatrix,
                    "distCoeffs": distCoeffs,
                    "rvec": rvec,
                    "tvec": tvec}
            return pose
        else:
            Warning("PnP solve failed, Pose return None.")
            return None


class charucoDiamond(charucoBoard):
    def __init__(self, ids=None, dictionary_id=cv2.aruco.DICT_6X6_50, marker_square_rate=0.6):
        super().__init__(dictionary_id, (3, 3), marker_square_rate, ids)
        self.marker_square_rate = marker_square_rate

    def detect(self, image):
        diamondCorners, diamondIds, markerCorners, markerIds = self.charuco_detector.detectDiamonds(
            image)
        return diamondCorners, diamondIds, markerCorners, markerIds

    def draw(self, image, diamondCorners, diamondIds, markerCorners, markerIds, pose=None, diamond_color=(255, 0, 0)):
        image_copy = charucoBoard().draw(image, markerCorners, markerIds, pose)
        image_copy = cv2.aruco.drawDetectedDiamonds(
            image_copy, diamondCorners, diamondIds, diamond_color)
        return image_copy
