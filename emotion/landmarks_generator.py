import face_alignment
from skimage import io
import os
import numpy as np
import cv2


# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
#
# input_path = 'C:/Users/Zber/Desktop/SavedData/Joy_15/'
#
# # input = io.imread('C:/Users/Zber/Desktop/SavedData/Joy_15')
#
# preds = fa.get_landmarks_from_directory(input_path)


def last_3chars(x):
    """Function that aids at list sorting.
      Args:
        x: String name of files.
      Returns:
        A string of the last 8 characters of x.
    """

    return (x[-3:])


def get_label(filepath):
    """Returns the label for a given Session by reading its
        corresponding CSV label file.
      Args:
        filepath: String path to the CSV file.
      Returns:
        String label name if found, else a -1.
    """

    if os.path.exists(filepath) and os.listdir(filepath):
        g = open(filepath + str(os.listdir(filepath)[0]), 'r')
        label = g.readline().split('.')[0].replace(" ", "")
        return label
    else:
        return -1


def get_landmarks(face_image_dir, label_main_dir, landmark_dir):
    for subject in sorted(os.listdir(face_image_dir), key=last_3chars):
        for session in sorted(
                os.listdir(face_image_dir + str(subject)), key=last_3chars):

            if session == ".DS_Store":
                continue

            # get image dir
            image_dir = os.path.join(face_imag_dir, subject, session)

            # get landmark label
            label_path = label_main_dir + str(subject) + '/' + str(
                session) + '/'
            label = get_label(label_path)

            if label == -1:
                continue

            # make landmark dir for each emotion class
            if not os.path.exists(landmark_dir + str(label)):
                os.makedirs(landmark_dir + str(label))

            # face alignment detect landmark
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            pred_landmarks = fa.get_landmarks_from_directory(image_dir)

            lms = np.zeros((len(pred_landmarks), 68, 2))

            for index, key in enumerate(pred_landmarks.keys()):
                lm = pred_landmarks[key]
                lms[index] = lm[0]

                # image plot check landmark detector performance
                image = cv2.imread(key)
                for idx, landmark_px in enumerate(lm[0]):
                    x, y = landmark_px[0], landmark_px[1]
                    x, y = int(x), int(y)
                    px = (x, y)
                    if idx ==33:
                        cv2.circle(image, px, 2, (0, 0, 255), 2)
                    else:
                        cv2.circle(image, px, 2, (0, 255, 0), 2)

                cv2.imshow("landmark image", image)
                cv2.setWindowProperty('landmark image', cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(50)

            # save landmarks to numpy file
            npy_name = "{}_{}_landmarks".format(subject, session)

            # np.save(os.path.join(landmark_dir, str(label), npy_name), lms)
            print("Complete {} Generation.".format(npy_name))


def landmark_length_alignment(landmarks, length=20, num_landmarks=68, n_dim=2):
    l = landmarks.shape[0]
    landmark_algin = np.zeros((length, num_landmarks, n_dim))

    landamrk_static = landmarks[0]

    if l <= length:
        landmark_algin[-1 * l:] = landmarks
        for i in range(length - l):
            landmark_algin[i] = landamrk_static

    else:
        landmark_algin[:] = landmarks[-1 * length:]

    return landmark_algin


def length_alignment_landmark(landmark_dir, alignment_landmark_dir):
    for label in range(1, 8):
        landmark_label_dir = os.path.join(landmark_dir, str(label))

        if not os.path.exists(alignment_landmark_dir + str(label)):
            os.makedirs(os.path.join(alignment_landmark_dir, str(label)))

        for npy_path in os.listdir(landmark_label_dir):
            d = np.load(os.path.join(landmark_label_dir, npy_path))
            landmark_algin = landmark_length_alignment(d)
            np.save(os.path.join(alignment_landmark_dir, str(label), npy_path), landmark_algin)
            print("Complete {} Generation in {}.".format(npy_path, os.path.join(alignment_landmark_dir, str(label))))


if __name__ == "__main__":
    # alignment face dir
    # face_imag_dir = "C:/Users/Zber/Documents/GitHub/Emotion-FAN/data/face/ck_face/"
    face_imag_dir = "C:/Users/Zber/Desktop/ckplus_example/"
    landmark_dir = "G:/My Drive/mmWave/mmWave-Emotion/mmWave Vision Datasets/CK+/CK+/alignment_landmarks/"
    label_dir = "G:/My Drive/mmWave/mmWave-Emotion/mmWave Vision Datasets/CK+/CK+/Emotion/"
    # alignment_landmark_dir = "G:/My Drive/mmWave/mmWave-Emotion/mmWave Vision Datasets/CK+/CK+/alignment_landmarks_L30/"
    alignment_landmark_dir = "G:/My Drive/mmWave/mmWave-Emotion/mmWave Vision Datasets/CK+/CK+/alignment_landmarks_L20/"

    # length_alignment_landmark(landmark_dir, alignment_landmark_dir)

    get_landmarks(face_imag_dir, label_dir, landmark_dir)
