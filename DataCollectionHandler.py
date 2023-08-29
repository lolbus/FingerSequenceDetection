from metadata import DatasetMeta
import cv2
import json
import numpy as np
import os
import time

metadata = DatasetMeta()
output_dir = metadata.SAVE_VID_DATA_DIR
os.makedirs(output_dir, exist_ok=True)

def convert(o):
  if isinstance(o, np.float32): return float(o)
  raise TypeError

def save_data(statusHandler, mainHandler, landmarks, roiframe, personframe, vectors, capture_validity, idx, cam_index):
    print("saving... for cam", cam_index)

    # Base line info to save
    time_now = time.time()
    new_row = [idx] + metadata.operation_mode_to_true_label_dict[mainHandler.operation_mode] + [capture_validity, time_now, cam_index]
    mainHandler.data_df.loc[len(mainHandler.data_df)] = new_row
    mainHandler.data_df.to_csv(metadata.HOME_DIR + "data_info.csv", index=False)

    # Begin saving camera / model 1 or 2 content
    print(f'saving idx {idx}')
    if statusHandler.save_hand_landmark_json:
        f_n = f"{metadata.hand_landmark_json_folder}/{str(idx)}.json"
        with open(f_n, 'w') as f:
            json.dump(landmarks, f, default=convert)

    '''# Save Full ROI
    if statusHandler.save_fullroi and len(landmarks) > 0:
        f_n = f"{metadata.fullroi_folder}/{str(idx)}.png" if cam_index == 1 else f"{metadata.fullroi_folder_2}/{str(idx)}.png"
        processed_frame = statusHandler.processed_frame_1 if cam_index == 1 else statusHandler.processed_frame_2
        cv2.imwrite(f_n, processed_frame)
    '''

    # Save Full ROI
    if statusHandler.save_fullroi and len(landmarks) > 0:
        f_n = f"{metadata.fullroi_folder}/{str(idx)}.png" if cam_index == 1 else f"{metadata.fullroi_folder_2}/{str(idx)}.png"
        processed_frame = statusHandler.processed_frame_1 if cam_index == 1 else statusHandler.processed_frame_2
        cv2.imwrite(f_n, processed_frame)

    # Save only the cropped person region that was used to detect left hand/right hand and landmarks
    if statusHandler.save_personcrop and statusHandler.pax_counter == 1 and cam_index == 1 and not personframe is None:
        f_n = f"{metadata.personcrop_folder}/{str(idx)}.png"
        cv2.imwrite(f_n, personframe)

    # Save the raw frame that was used to detect person
    if statusHandler.save_rawroi and not roiframe is None:
        f_n = f"{metadata.rawroi_folder}/{str(idx)}.png" if cam_index == 1 else f"{metadata.rawroi_folder_2}/{str(idx)}.png"
        cv2.imwrite(f_n, roiframe)
        print('rawroi save success')
    else:
        print('no save')

    if statusHandler.save_vectors_info and cam_index == 1 and not vectors is None:
        f_n = f"{metadata.vectors_folder}/{str(idx)}.json"
        data_dict = {'finger_vectors': vectors, 'capture_validity': capture_validity}
        with open(f_n, 'w') as f:
            json.dump(data_dict, f)
    if mainHandler.record_mode == "automatic":
        time.sleep(3) # Sleep 2s slow down frame flows to reduce duplicate frames
    else:
        time.sleep(2)
    mainHandler.recorded_memory[cam_index - 1] = 1
    print("save complete, rm stat, cam_index", mainHandler.recorded_memory, cam_index)


    # SAVE DATA CODE ENDS HERE ---