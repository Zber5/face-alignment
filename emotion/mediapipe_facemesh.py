import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import time
from pathlib import Path
import numpy as np
import math
from typing import List, Mapping, Optional, Tuple, Union
from imutils.video import count_frames
import os
import enum
from emotion.key_landmark import get_face_landmark_style, get_key_flm, BLACK_COLOR

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

total_FLms = 468
frame_id = 0
camera_fps = 30
width = 1280
height = 720
dims = 2


# def render_color
def get_num_flm():
    pos = []
    start = 0
    key_flm = get_key_flm()
    for key in key_flm:
        length = len(key_flm[key])
        end = start + length
        pos.append((start, end))
        start = end
    return start, pos


def flm_score(all_flm):
    num_flm, flm_seg = get_num_flm()
    pre = None
    key_flm = get_key_flm()

    all_dis = np.zeros((all_flm.shape[0] - 1, num_flm))

    for fid, flm in enumerate(all_flm):
        cur = np.zeros((num_flm, dims))
        for key, s_e in zip(key_flm, flm_seg):
            indices = np.array(key_flm[key])
            start, end = s_e
            pos = flm[indices]
            cur[start:end] = pos
        if pre is not None:
            # calculate the score
            dist = np.linalg.norm(cur - pre, axis=1)
            all_dis[fid - 1] = dist
        pre = cur

    return all_dis


def key_average_score(scores, num_frame=89):
    num_flm, flm_seg = get_num_flm()
    num_keys = len(flm_seg)
    data = np.zeros((num_frame, num_keys))
    for i in range(num_keys):
        s_ind, e_ind = flm_seg[i]
        flmscore = scores[:, s_ind:e_ind]
        mean = np.mean(flmscore, axis=1)
        length = np.shape(flmscore)[0]
        if length <= num_frame:
            data[:length, i] = mean
        else:
            data[:num_frame, i] = mean
    return data


def flm_detector(video_path, output_path, output_as_video=False, output_flm_video=False, output_flm_npy=False, dim=2):
    cv_plot = False

    # face mesh settings
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    # drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
    drawSpec = get_face_landmark_style()
    black_drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
    connection_drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    frame_id = 0
    pTime = 0

    cap = cv2.VideoCapture(video_path)
    total_frame = count_frames(video_path)
    all_FLms = np.zeros((total_frame, total_FLms, dim))
    if output_as_video or output_flm_video:
        # video_output = os.path.join(output_path,
        #                             "{}_landmark_flm.avi".format(os.path.basename(os.path.normpath(output_path))))
        video_output = os.path.join(output_path,
                                    "{}_landmark.avi".format(os.path.basename(os.path.normpath(output_path))))
        # fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        video_writer = cv2.VideoWriter(str(video_output),
                                       fourcc,
                                       camera_fps,
                                       (width, height))

    while True:
        success, img = cap.read()
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not success:
            # print("Finish")
            break
        imgRGB = img
        results = faceMesh.process(imgRGB)
        black_img = np.zeros(img.shape)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, connection_drawSpec)
                mpDraw.draw_landmarks(img, faceLms, None, drawSpec, connection_drawSpec)
                mpDraw.draw_landmarks(black_img, faceLms, None, black_drawSpec, None)

            np_faceLms = np.zeros((total_FLms, 2))
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = _normalized_to_pixel_coordinates(lm.x, lm.y, iw, ih)
                # x,y = int(lm.x*iw), int(lm.y*ih)
                # print(id,x,y)
                np_faceLms[id] = [x, y]

            all_FLms[frame_id] = np_faceLms

        # cv ploting
        if cv_plot:
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(black_img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            WindowName = "Image"
            cv2.imshow("Image", black_img)
            cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(10)

        if output_flm_video:
            # b_img = np.copy(black_img)
            # rgb_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2RGB)
            video_writer.write(black_img.astype('uint8'))
        elif output_as_video:
            # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_writer.write(img)

        frame_id += 1

    cap.release()

    if output_as_video or output_flm_video:
        # Close the video writer
        video_writer.release()


    if output_flm_npy:
        flm_npy_path = os.path.join(output_path,
                                    "{}_flm".format(os.path.basename(os.path.normpath(output_path))))
        np.save(flm_npy_path, all_FLms)

    return all_FLms


def color_map_generator_rgb(all_FLms, ih=720, iw=1080):
    # image_rows, image_cols, _ = image.shape
    # brg_channel =
    # heatmap = np.zeros((total_frame, ih, iw))
    num_channel = 3
    red = 2
    total_frame = all_FLms.shape[0]
    f_heatmap = np.zeros((total_frame, ih, iw, num_channel))
    heatmap = np.zeros((ih, iw, num_channel))

    for f_idx in range(1, all_FLms.shape[0] - 1):
        prev = all_FLms[f_idx - 1]
        cur = all_FLms[f_idx]
        dist = np.linalg.norm(cur - prev, axis=1)

        for lm_idx in range(total_FLms):
            h, w = cur[lm_idx]
            # heatmap[f_idx, int(w), int(h)] = heatmap[f_idx - 1, int(w), int(h)] + dist[lm_idx]
            heatmap[int(w), int(h), red] = heatmap[int(w), int(h), red] + dist[lm_idx]

        img = heatmap
        WindowName = "Heatmap"
        cv2.imshow("Heatmap", img)
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(30)
        f_heatmap[f_idx] = heatmap
    print("")


def color_map_generator(all_FLms, output_path, output_as_video=False):
    total_frame = all_FLms.shape[0]
    f_heatmap = np.zeros((total_frame, height, width))
    heatmap = np.zeros((height, width))

    if output_as_video:
        video_output = os.path.join(output_path,
                                    "{}_heatmap.avi".format(os.path.basename(os.path.normpath(output_path))))
        fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        video_writer = cv2.VideoWriter(str(video_output),
                                       fourcc,
                                       camera_fps,
                                       (width, height))

    for f_idx in range(1, all_FLms.shape[0] - 1):
        prev = all_FLms[f_idx - 1]
        cur = all_FLms[f_idx]
        dist = np.linalg.norm(cur - prev, axis=1)

        for lm_idx in range(total_FLms):
            h, w = cur[lm_idx]
            # heatmap[f_idx, int(w), int(h)] = heatmap[f_idx - 1, int(w), int(h)] + dist[lm_idx]
            heatmap[int(w), int(h)] = heatmap[int(w), int(h)] + dist[lm_idx]

        heatmapshow = None
        heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        WindowName = "Heatmap"
        cv2.imshow(WindowName, heatmapshow)
        if output_as_video:
            # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_writer.write(heatmapshow)
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(20)

        # save to numpy
        f_heatmap[f_idx] = heatmap

    if output_as_video:
        # Close the video writer
        video_writer.release()

    return f_heatmap


#
# def color_map_generator2(all_FLms, ih=720, iw=1080):
#     # image_rows, image_cols, _ = image.shape
#     # brg_channel =
#     heatmap = np.zeros((total_frame, ih, iw))
#     # heatmap = np.zeros((ih, iw))
#
#     for f_idx in range(1, all_FLms.shape[0] - 1):
#         prev = all_FLms[f_idx - 1]
#         cur = all_FLms[f_idx]
#         dist = np.linalg.norm(cur - prev, axis=1)
#
#         for lm_idx in range(total_FLms):
#             h, w = cur[lm_idx]
#             heatmap[f_idx] = heatmap[f_idx-1]
#             heatmap[f_idx, int(w), int(h)] = heatmap[f_idx, int(w), int(h)] + dist[lm_idx]
#
#
#         # cv2.applyColorMap(heatmap[f_idx], cv2.COLORMAP_JET)
#
#     # normalize
#     # heatmap = heatmap / np.max(heatmap)
#
#     for f_idx in range(total_frame):
#         img = heatmap[f_idx]
#         WindowName = "Heatmap"
#         cv2.imshow("Heatmap", img)
#         cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
#         cv2.waitKey(30)


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


if __name__ == "__main__":
    # video_path = "C:/Users/Zber/Desktop/SavedData_MIMO/Anger_1/Anger_1.avi"
    video_path = "C:/Users/Zber/Desktop/Subjects_Video/S1/Anger_1.avi"

    root_path = "C:/Users/Zber/Desktop/SavedData_MIMO/"
    # output_data_path = "C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/"
    output_data_path = "C:/Users/Zber/Desktop/Subjects_Video/S1/"
    file_prefix = "Anger_1_flm"

    video_dir = "{}_{}"
    video_name = "{}_{}.avi"
    npy_name = "{}_{}"

    # start index
    start_index = 0
    end_index = 1
    # emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    emotion_list = ['Joy']
    num_records = (end_index - start_index) * len(emotion_list)
    num_frame = 89
    save_npy = False

    _, keyparts = get_num_flm()

    # data
    score_data = np.zeros((num_records, num_frame, len(keyparts)))

    index = 0

    for l, e in enumerate(emotion_list):
        for i in range(start_index, end_index):
            video_path = os.path.join(root_path, video_dir.format(e, i), video_name.format(e, i))
            output_path = os.path.join(root_path, video_dir.format(e, i))
            all_FLms = flm_detector(video_path, output_path, output_as_video=True, output_flm_video=False)
            key_score = flm_score(all_FLms)

            # average data
            avg_score = key_average_score(key_score)

            score_data[i] = avg_score

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
