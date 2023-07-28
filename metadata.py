import numpy as np


class DatasetMeta(object):
    """metadata class which store all prefix meta values related to the dataset"""

    def __init__(self):
        # Data collection configs:
        self.modelDir = "C:/Users/ASUS/Desktop/IVOD_Models/"  # Directory of saved weights checkpoint
        self.mmPoseModelDir = "C:/Users/ASUS/Desktop/PoseMM/"
        self.HOME_DIR = "C:/Users/ASUS/Desktop/22June_DataCollection/"
        self.SAVE_VID_DATA_DIR = self.HOME_DIR + "(VID)PassengerNo_0 (Empty)"
        self.person_detector_threshold = 0.2
        self.LRhand_detector_threshold_dict = {0: 0.3, 1: 0.3, 2: 0.15, 3: 0.25, 4: 0.25, 5: 0.1, 6: 0.05}
        self.hand_detector_threshold_dict = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.0}
        self.hand_landmark_json_folder = self.HOME_DIR + "HAND_LANDMARKS_JSON"
        self.fullroi_folder = self.HOME_DIR + "FULL_ROI"  # Full ROI. Everything detected
        self.personcrop_folder = self.HOME_DIR + "PERSON_CROP"  # Crop of the blue rectangle in the frame (The person capture)
        self.rawroi_folder = self.HOME_DIR + "RAW_ROI"  # No Landmark points retained
        self.rawpersoncrop_folder = self.HOME_DIR + "RAW_PERSON_CROP"  # Crop of the blue rectangle in the frame (The person capture)
        self.vectors_folder = self.HOME_DIR + "VECTORS_JSON" # Predicted angles and validation bool of manual check

        self.operation_mode_xy_offset = {0: (-30, -30), 1: (-50, -40), 2: (-150, -150), 3: (-30, -40), 4: (-50, -40),
                                         5: (-150, -150), 6: (-300, -300)}
        self.operation_mode_widthheight_offset = {0: (0, 0), 1: (60, 30), 2: (90, 75), 3: (10, 40), 4: (20, 20),
                                         5: (90, 75), 6: (300, 300)}

        a = np.array([0, 60])  # coordinates of the first vertex
        b = np.array([96, 14])  # coordinates of the second vertex
        c = np.array([196, 87])  # coordinates of the third vertex
        d = np.array([128, 156])  # coordinates of the fourth vertex

        self.finger_reader_quadrilateral = (a, b, c, d)  # coordinates of the first vertex

        self.operation_mode_to_true_label_dict = {0: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                  1: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                  2: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                  3: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                  4: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                  5: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                                  6: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]}
        self.operation_mode_to_hand_foi_dict = {0: ((1, 0), "Thumb"),
                                                1: ((1, 0), "Index"),
                                                2: ((1, 0), "Index", "Middle", "Ring", "Pinky"),
                                                3: ((0, 1), "Thumb"),
                                                4: ((0, 1), "Index"),
                                                5: ((0, 1), "Index", "Middle", "Ring", "Pinky"),
                                                6: ((1, 1), "Thumb")}
