from metadata import DatasetMeta
import cv2
import json
import numpy as np
import os

metadata = DatasetMeta()
output_dir = metadata.SAVE_VID_DATA_DIR
os.makedirs(output_dir, exist_ok=True)

def convert(o):
  if isinstance(o, np.float32): return float(o)
  raise TypeError

def save_data(statusHandler, landmarks, roiframe, personframe, vectors, capture_validity, idx):
    if statusHandler.save_hand_landmark_json:
        f_n = f"{metadata.hand_landmark_json_folder}/{str(idx)}.json"
        with open(f_n, 'w') as f:
            json.dump(landmarks, f, default=convert)

    # Save Full ROI
    if statusHandler.save_fullroi and len(landmarks['visualization']) > 0:
        f_n = f"{metadata.fullroi_folder}/{str(idx)}.png"
        cv2.imwrite(f_n, statusHandler.processed_frame)

    # Save only the cropped person region that was used to detect left hand/right hand and landmarks
    if statusHandler.save_personcrop and statusHandler.pax_counter == 1:
        f_n = f"{metadata.personcrop_folder}/{str(idx)}.png"
        cv2.imwrite(f_n, personframe)

    # Save the raw frame that was used to detect person
    if statusHandler.save_rawroi and statusHandler.pax_counter == 1:
        f_n = f"{metadata.rawroi_folder}/{str(idx)}.png"
        cv2.imwrite(f_n, roiframe)

    if statusHandler.save_vectors_info:
        f_n = f"{metadata.vectors_folder}/{str(idx)}.json"
        data_dict = {'finger_vectors': vectors, 'capture_validity': capture_validity}
        with open(f_n, 'w') as f:
            json.dump(data_dict, f)


    # SAVE DATA CODE ENDS HERE ---