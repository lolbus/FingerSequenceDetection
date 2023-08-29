import numpy as np


class DatasetMeta(object):
    """metadata class which store all prefix meta values related to the dataset"""

    def __init__(self):
        # Data collection configs:
        self.mmPoseModelDir = "C:/Users/ASUS/Desktop/DDA_Exercise/PoseMM_Pretrained_models/"
        self.HOME_DIR = "C:/Users/ASUS/Desktop/22June_DataCollection/"
        self.SAVE_VID_DATA_DIR = self.HOME_DIR + "(VID)PassengerNo_0 (Empty)"
        self.person_detector_threshold = 0.4
        self.LRhand_detector_threshold_dict = {0: 0.3, 1: 0.3, 2: 0.15, 3: 0.25, 4: 0.25, 5: 0.1, 6: 0.05}
        self.hand_detector_threshold_dict = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.0}
        self.hand_landmark_json_folder = self.HOME_DIR + "HAND_LANDMARKS_JSON"
        self.fullroi_folder = self.HOME_DIR + "FULL_ROI"  # Full ROI. Everything detected
        self.fullroi_folder_2 = self.HOME_DIR + "FULL_ROI2"  # Full ROI. Everything detected
        self.personcrop_folder = self.HOME_DIR + "PERSON_CROP"  # Crop of the blue rectangle in the frame (The person capture)
        self.rawroi_folder = self.HOME_DIR + "RAW_ROI"  # No Landmark points retained
        self.rawroi_folder_2 = self.HOME_DIR + "RAW_ROI2"  # No Landmark points retained
        self.rawpersoncrop_folder = self.HOME_DIR + "RAW_PERSON_CROP"  # Crop of the blue rectangle in the frame (The person capture)
        self.vectors_folder = self.HOME_DIR + "VECTORS_JSON" # Predicted angles and validation bool of manual check

        self.operation_mode_xy_offset = {0: (-30, -30), 1: (-50, -40), 2: (-150, -150), 3: (-30, -40), 4: (-50, -40),
                                         5: (-150, -150), 6: (-300, -300)}
        self.operation_mode_widthheight_offset = {0: (150, 150), 1: (60, 30), 2: (90, 75), 3: (60, 80), 4: (20, 20),
                                         5: (90, 75), 6: (300, 300)}

        # Cam 1 green box boundary points
        a = np.array([0, 60])  # coordinates of the first vertex
        b = np.array([96, 14])  # coordinates of the second vertex
        c = np.array([196, 87])  # coordinates of the third vertex
        d = np.array([128, 156])  # coordinates of the fourth vertex

        # Cam 2 green box boundary points
        e = np.array([45, 184])  # coordinates of the first vertex
        f = np.array([245, 180])  # coordinates of the second vertex
        g = np.array([282, 312])  # coordinates of the third vertex
        h = np.array([4, 319])  # coordinates of the fourth vertex


        self.finger_reader_quadrilateral_1 = (a, b, c, d)  # coordinates of the first vertex
        self.finger_reader_quadrilateral_2 = (e, f, g, h)  # coordinates of the 2nd vertex

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

        # Keypoint score must be greater than threshold to activate verification check if non foi is inside scanner, if confident score is low and prediction base on guess of existence of non foi,
        # we can use the threshold score to ignore poor visibility non foi
        self.KEYPOINT_SCORES_THRESHOLD_DICT = {
            'default': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'camindex2 operationmode0': [0.1, 0.1, 0.1, 0.1, 0.1, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44,
                                         0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44],
            'camindex2 operationmode3': [0.1, 0.1, 0.1, 0.1, 0.1, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44,
                                         0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44],
            'camindex2 operationmode1': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'camindex2 operationmode4': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'camindex2 operationmode2': [0.44, 0.44, 0.44, 0.44, 0.44, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                                         0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            'camindex2 operationmode5': [0.44, 0.44, 0.44, 0.44, 0.44, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                                         0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        }

        self.det_inferencer = {'FASTERRCNN_LH_DETECTOR_V1':{'CONFIG': self.mmPoseModelDir + '/faster-rcnn_r50_fpn_1x_coco.py',
                                                            'CHECKPOINT': self.mmPoseModelDir + '/faster-rcnn_r50_fpn_1x_coco_030823_v1.pth'},
                               'FASTERRCNN_LH_DETECTOR_V2': {
                                   'CONFIG': self.mmPoseModelDir + '/faster-rcnn_r50_fpn_1x_coco_v2.py',
                                   'CHECKPOINT': self.mmPoseModelDir + '/faster-rcnn_r50_fpn_1x_coco_100823_v2.pth'},
                               'FASTERRCNN_LH_DETECTOR_V3': {
                                   'CONFIG': self.mmPoseModelDir + '/faster-rcnn_r50_fpn_1x_coco_v3_15aug2119.py',
                                   'CHECKPOINT': self.mmPoseModelDir + '/faster-rcnn_r50_fpn_1x_coco_v3_15aug2119.pth'},
                               'FASTERRCNN_FORMATION_DETECTOR_V1': {
                                   'CONFIG': self.mmPoseModelDir + '/faster-rcnn-21aug23-first_top_view_g.py',
                                   'CHECKPOINT': self.mmPoseModelDir + '/faster-rcnn-21aug23-first_top_view_g.pth'},
                               }
