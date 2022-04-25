import sys
import numpy as np
import os
from feat import Detector
import cv2
import time
from IPython.display import Video

face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "JAANET"
# au_model = "DRML"
# au_model = "rf"
# au_model = "svm"
# au_model = "logistic"
emotion_model = "resmasknet"

detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model,
                    emotion_model=emotion_model)

au20index = [
    f"AU{str(i).zfill(2)}"
    for i in [
        1,
        2,
        4,
        5,
        6,
        7,
        9,
        10,
        12,
        14,
        15,
        17,
        18,
        20,
        23,
        24,
        25,
        26,
        28,
        43,
    ]
]


def au_extractor(video_path):
    video_prediction = detector.detect_video(video_path, skip_frames=1)
    # video_prediction.loc[[2]].plot_detections()
    # video_prediction.loc[[80]].plot_detections()
    aus = video_prediction.aus()
    aus_npy = aus.to_numpy()
    return aus_npy


# video_path = "C:/Users/Zber/Desktop/SavedData_MIMO/Anger_1/Anger_1.avi"
# video_prediction = detector.detect_video(video_path, skip_frames=1)

# video_prediction.aus()

# select_video_prediction = video_prediction[au20index]

# video_prediction.loc[[48]].plot_detections()
# video_prediction.emotions().plot()
# cap = cv2.VideoCapture(video_path)
#
#
# while True:
#     success, img = cap.read()
#     # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     if not success:
#         # print("Finish")
#         break
#     imgRGB = img
#     results = detector.detect_image(imgRGB)
#     results.plot_detections()
#     time.sleep(0.5)
if __name__ == "__main__":

    # video_path = "C:/Users/Zber/Desktop/SavedData_MIMO/Anger_1/Anger_1.avi"
    root_path = "C:/Users/Zber/Desktop/SavedData_MIMO/"
    output_data_path = "C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/"
    file_prefix = "aus_jannet_s40_e80"

    video_dir = "{}_{}"
    video_name = "{}_{}.avi"
    npy_name = "{}_{}"

    # start index
    start_index = 40
    end_index = 80
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    num_records = (end_index - start_index) * len(emotion_list)
    num_frame = 89
    num_aus = 12
    # num_aus = 20
    save_npy = True

    # data
    score_data = np.zeros((num_records, num_frame, num_aus))

    index = 0

    for l, e in enumerate(emotion_list):
        for i in range(start_index, end_index):
            video_path = os.path.join(root_path, video_dir.format(e, i), video_name.format(e, i))
            aus = au_extractor(video_path)
            length = np.shape(aus)[0]
            if length <= num_frame:
                score_data[index, :length] = aus
            else:
                score_data[index, :] = aus[:num_frame, :]

            # if save_npy:
            #     np.save(save_path, key_score)

            # color_map_generator_rgb(all_FLms)
            # color_map_generator(all_FLms, output_path, output_as_video=True)
            index += 1
            print("{} Complete".format(video_name.format(e, i)))

    # save npy file
    if save_npy:
        save_path = os.path.join(output_data_path, file_prefix)
        np.save(save_path, score_data)
        print("Npy file saved")

    # video_path = "C:/Users/Zber/Desktop/SavedData_MIMO/Anger_1/Anger_1.avi"
    # video_prediction = detector.detect_video(video_path, skip_frames=1)
    # video_prediction.loc[[48]].plot_detections()
    # video_prediction.emotions().plot()
    # print("")
