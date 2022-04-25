import os, sys, random
import matplotlib.pyplot as plt
from shutil import copyfile
import numpy as np
# from mtcnn.mtcnn import MTCNN
from scipy import misc
import cv2
import time
import shutil

import mediapipe as mp


def last_8chars(x):
    """Function that aids at list sorting.
      Args:
        x: String name of files.
      Returns:
        A string of the last 8 characters of x.
    """

    return (x[-18:-10])


def last_8chars_img(x):
    """Function that aids at list sorting.
      Args:
        x: String name of files.
      Returns:
        A string of the last 8 characters of x.
    """

    return (x[-8:])


def last_3chars(x):
    """Function that aids at list sorting.
      Args:
        x: String name of files.
      Returns:
        A string of the last 8 characters of x.
    """

    return (x[-3:])


def get_vid_name(file):
    """Creates the new video name.
      Args:
        file: String name of the original image name.
      Returns:
        A string name defined by the Subject and Session
        numbers.
    """

    return file.split("_")[0] + "_" + file.split("_")[1]


def create_moving_batch(images_list, depth):
    """Creates the list of neutral and emotion-labeled images to be processed.
        From the whole list of images of a given session, create an 2 dimensional
        array with neutral and emotion-labeled image names. The quantity of
        images per index in batch (same for neutral and emotion) is defined by
        the selected clip depth. From the sequence, the first depth number of
        images are placed at batch[1], and the last depth number of images at
        batch[0].
      Args:
        images_list: String array with paths to all images from a given session.
        depth: Integer number of images to be appended in each batch index. It
            must be at most half of the total session quantity of images.
      Returns:
        A 2D string array containing the path to the images to be processed later
        to create 3D arrays:
            batch[0] = emotion-labeled images
            batch[1] = neutral-labeled images
    """

    batch = [[], []]
    for index, image in enumerate(sorted(images_list, key=last_8chars)):
        if (index < depth):
            batch[1] = np.append(batch[1], image)
        if (index > len(images_list) - depth - 1):
            batch[0] = np.append(batch[0], image)
    return batch


def move_images(images_list, src, dest, neu_dest, min_seq_len, depth, t_height,
                t_width, crop_faces_flag, detector):
    """Creates the Numpy binary files with a 3D array of a sequence of images.
      Args:
        images_list: String array with paths to all images from a given Session.
        src: String path to a selected images Session.
        dest: String path to where the emotion-labeled Numpy binary files must
            be saved.
        neu_dest: String path to where the neutral-labeled Numpy binary files
        must be saved.
        min_seq_len: Integer number of minimum images in a given Session for them
            to be processed. Only Sessions with more than this value will be
            considered.
        depth: Integer number of images to be appended to create a 3D array.
            This is the temporal depth of the input instances.
        t_height: Integer number of the target height to which the images will
            be resized.
        t_width: Integer number of the target width to which the images will
            be resized.
        crop_faces_flag: Boolean value that indicates if facial cropping will be
            done before resizing the original images.
        detector: MTCNN object.
    """

    # Create neutral and emotion-labeled lists to be processed.
    batch = create_moving_batch(images_list, depth)

    # Emotion batch
    vid_name = get_vid_name(batch[0][0])
    if not os.path.exists(dest + '/' + vid_name):
        im_size = cv2.imread(src + batch[0][0]).shape

        im_shape = im_size
        vid = np.zeros(0)
        for index, image in enumerate(batch[0]):
            if crop_faces_flag:
                im = crop_face(src + image, detector)
            else:
                im = cv2.imread(src + image)
            im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)
            if index == 0:
                temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                 im_resized.shape[2]))
            temp = np.append(temp, [im_resized], axis=0)
            vid = temp
        # Save to Numpy binary file
        try:
            np.save(dest + '/' + vid_name, vid)
        except Exception as e:
            print("Unable to save instance at:",
                  dest + str(vid_name[0]) + "_" + str(vid_name[1]))
            raise

    # Neutral batch
    vid_name = get_vid_name(batch[1][0])
    if not os.path.exists(neu_dest + '/' + vid_name):
        im_size = cv2.imread(src + batch[1][0]).shape
        vid = np.zeros(0)
        for index, image in enumerate(batch[1]):
            if crop_faces_flag:
                im = crop_face(src + image, detector)
            else:
                im = cv2.imread(src + image)
            im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)
            if index == 0:
                temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                 im_resized.shape[2]))
            temp = np.append(temp, [im_resized], axis=0)
            vid = temp
        # Save to Numpy binary file
        try:
            np.save(neu_dest + '/' + vid_name, vid)
        except Exception as e:
            print("Unable to save instance at:",
                  dest + str(vid_name[0]) + "_" + str(vid_name[1]))
            raise


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


def get_landmarks(lm_main_dir,subject, session, filepath_list, num_landmarks=68, n_dim=2):
    landmark_npy = np.zeros((len(filepath_list), num_landmarks, n_dim))

    for index, filepath in enumerate(filepath_list):

        filepath = os.path.join(lm_main_dir, subject, session, filepath)

        if os.path.isfile(filepath):
            landmark = np.loadtxt(filepath)
            landmark_npy[index] = landmark
        else:
            raise Exception("non-existent landmark files")

    return landmark_npy





def landmark_length_alignment(landmarks, length=30, num_landmarks=68, n_dim=2):
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


# def crop_face(image_path, detector):
#     """Returns a facial image if face is detected.
#
#         If the MTCNN face detector finds a face in the image, the returned
#         image size will be defined by the bounding box returned by the face
#         detector. Otherwise, the imagewill be centered cropped by empirically
#         found parameters.
#       Args:
#         image_path: String path to the image file.
#         detector: MTCNN object
#       Returns:
#         OpenCV image object of the processed image.
#     """
#
#     image = cv2.imread(image_path)
#     if detector:
#         result = detector.detect_faces(image)
#         if result:
#             bbox = result[0]['box']
#             image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:
#                                                      bbox[0] + bbox[2], :]
#         else:
#             print("Face could not be detected in image",
#                   image_path + ", " + "proceeding with center cropping.")
#             image = image[:448, 96:-96, :]
#     return image


def crop_face(image_path, detector):
    """Returns a facial image if face is detected.

        If the MTCNN face detector finds a face in the image, the returned
        image size will be defined by the bounding box returned by the face
        detector. Otherwise, the imagewill be centered cropped by empirically
        found parameters.
      Args:
        image_path: String path to the image file.
        detector: MTCNN object
      Returns:
        OpenCV image object of the processed image.
    """

    image = cv2.imread(image_path)
    if detector:
        result = detector.detectMultiScale(image, 1.1, 4)
        if result is not None:
            (x, y, w, h) = result[0]
            image = image[y:y + h, x:x + w, :]
        else:
            print("Face could not be detected in image",
                  image_path + ", " + "proceeding with center cropping.")
            image = image[:448, 96:-96, :]
    return image


def gen_emotionsFolders(label_main_dir, lm_main_dir, emotions_dir,
                        neutral_label, min_seq_len, length):
    """Generates a folder of Numpy binary files with 3D arrays of images for each
        label in CK+ database.
        Check each Session if the number of images is more than min_seq_len and
        create a 3D array by appending a depth number of images together. Each
        image is resized to (t_height, t_width) from its original size or from
        a face bounding box if the flag crop_faces is true. Numpy binary files
        are saved in emotions_dir.
      Args:
        main_dir: String path to where the labels and images folders are placed.
        label_main_dir: String path to CK+ labels folder
        image_main_dir: String path to CK+ images folder
        emotions_dir: String path where to Numpy binary files will be saved.
        crop_faces_flag: Boolean value that indicates if facial cropping will be
            done before resizing the original images.
        neutral_label: String name of the neutral label. This value has to be
            defined by the user because it will not appear in the CSV files.
        min_seq_len: Integer number of minimum images in a given Session for them
            to be processed. Only Sessions with more than this value will be
            considered.
        depth: Integer number of images to be appended to create a 3D array.
            This is the temporal depth of the input instances.
        t_height: Integer number of the target height to which the images will
            be resized.
        t_width: Integer number of the target width to which the images will
            be resized.
    """

    # try:
    #     assert (min_seq_len >= depth * 2)
    # except AssertionError as e:
    #     print('The depth should be at most equal to half of min_seq_len.')
    #     raise

    start_time = time.time()
    print("Getting Landmarks...")

    # if crop_faces == "True" or crop_faces == "1":
    #     detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    # else:
    #     detector = 0

    if os.path.exists(emotions_dir):
        shutil.rmtree(emotions_dir, ignore_errors=True)

    list_labels = np.array([])
    list_labels = np.append(list_labels, neutral_label)
    if not os.path.exists(emotions_dir + neutral_label):
        os.makedirs(emotions_dir + str(neutral_label))
    for subject in sorted(os.listdir(lm_main_dir), key=last_3chars):
        for session in sorted(
                os.listdir(lm_main_dir + str(subject)), key=last_3chars):
            if session != ".DS_Store":
                landmarks_path = lm_main_dir + str(subject) + '/' + str(
                    session) + '/'
                landmarks_list = [
                    x
                    for x in sorted(os.listdir(landmarks_path), key=last_8chars)
                    if x.split(".")[1] == "txt"
                ]
                if (landmarks_list and len(landmarks_list) >= min_seq_len):
                    label_path = label_main_dir + str(subject) + '/' + str(
                        session) + '/'
                    label = get_label(label_path)
                    landmarks = get_landmarks(lm_main_dir,subject, session, landmarks_list)
                    landmarks_align = landmark_length_alignment(landmarks, length=length)
                    if label != -1:
                        if label not in list_labels:
                            list_labels = np.append(list_labels, label)
                            if not os.path.exists(emotions_dir + str(label)):
                                os.makedirs(emotions_dir + str(label))

                        # save numpy file
                        np.save(emotions_dir + str(label) + '/' + get_vid_name(landmarks_list[0]) + "_landmarks",
                                landmarks_align)

    duration = time.time() - start_time
    print("\nDone! Total time %.1f seconds." % (duration))
    test_vid = np.load(emotions_dir + neutral_label + "/" +
                       sorted(os.listdir(emotions_dir + neutral_label))[0])
    print("Clip size:", test_vid.shape)
    print("Neutral label example:")
    plt.imshow(test_vid[0][:, :, 0], cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.show()


def main(argv):
    gen_emotionsFolders(
        label_main_dir=str(argv[0]),
        lm_main_dir=str(argv[5]),
        emotions_dir=str(argv[1]),
        neutral_label=str(argv[2]),
        min_seq_len=int(argv[3]),
        length=int(argv[4]))


def check_data(landmark_dir, image_main_dir):
    splits = landmark_dir.split('/')
    subject = splits[-2]
    session = splits[-1]

    image_dir = os.path.join(image_main_dir, subject, session)

    mpDraw = mp.solutions.drawing_utils
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    for lm_name, image_name in  zip(sorted(os.listdir(landmark_dir), key=last_8chars),
                                sorted(os.listdir(image_dir), key=last_8chars_img)):

        landmark_path = os.path.join(landmark_dir, lm_name)
        image_path = os.path.join(image_dir, image_name)

        landmark = np.loadtxt(landmark_path)
        image = cv2.imread(image_path)

        for idx, landmark_px in enumerate(landmark):
            x, y = landmark_px[0], landmark_px[1]
            x, y = int(x), int(y)
            px = (x,y)
            drawing_spec = drawSpec
            cv2.circle(image, px, drawing_spec.circle_radius,
                       drawing_spec.color, drawing_spec.thickness)

        cv2.imshow("landmark image", image)
        cv2.setWindowProperty('landmark image', cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(50)


def check_landmark(landmark_path, image_main_dir, length=30):
    splits = landmark_path[-22:].split('_')
    subject = splits[-3]
    session = splits[-2]

    image_dir = os.path.join(image_main_dir, subject, session)

    mpDraw = mp.solutions.drawing_utils
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    image_path_list = sorted(os.listdir(image_dir), key=last_8chars_img)
    first_image_path = image_path_list[0]

    if len(image_path_list)<=length:
        for i in range(length-len(image_path_list)):
            image_path_list.insert(0, first_image_path)
    else:
        image_path_list = image_path_list[-30:]

    landmarks = np.load(landmark_path)

    for landmark, image_name in  zip(landmarks, image_path_list):

        image_path = os.path.join(image_dir, image_name)

        image = cv2.imread(image_path)

        for idx, landmark_px in enumerate(landmark):
            x, y = landmark_px[0], landmark_px[1]
            x, y = int(x), int(y)
            px = (x,y)
            drawing_spec = drawSpec
            cv2.circle(image, px, 3,
                       drawing_spec.color, 3)

        cv2.imshow("landmark image", image)
        cv2.setWindowProperty('landmark image', cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(100)


def count_classes(landmark_dir):
    emotion_classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    total_num = []

    for i in range(1,8):
        label_folder = os.path.join(landmark_dir, str(i))
        npy_list = os.listdir(label_folder)
        print("{} has {} records".format(emotion_classes[i-1], len(npy_list)))
        total_num.append(len(npy_list))

    print("total number is {}".format(np.sum(total_num)))







if __name__ == "__main__":


    main_dir = "G:\My Drive\mmWave\mmWave-Emotion\mmWave Vision Datasets\CK+\CK+".replace("\\", "/")

    landmark_dir = os.path.join(main_dir, "Landmarks/S005/001")
    # landmark_path = os.path.join(main_dir, "emotion_images/1/S010_004_landmarks.npy")
    # landmark_path = os.path.join(main_dir, "emotion_images/3/S005_001_landmarks.npy")
    landmark_path = os.path.join(main_dir, "emotion_images/3/S098_003_landmarks.npy")
    generated_dir = os.path.join(main_dir, "emotion_images/")

    labels_main_dir = os.path.join(main_dir, "Emotion/")
    image_main_dir = os.path.join(main_dir, "cohn-kanade-images/")
    lm_main_dir = os.path.join(main_dir, "Landmarks/")
    emotions_dir = os.path.join(main_dir, "emotion_images/")

    alignment_landmark_dir = "G:/My Drive/mmWave/mmWave-Emotion/mmWave Vision Datasets/CK+/CK+/alignment_landmarks_L30/"

    # check_data(landmark_dir, image_main_dir)

    check_landmark(landmark_path, image_main_dir)
    #
    # count_classes(alignment_landmark_dir)

    # crop_faces = 1
    # neutral_label = 0
    # min_seq_len = 5
    # depth = 5
    # t_height = 300
    # t_width = 300
    # length = 30
    # main([labels_main_dir, emotions_dir, neutral_label, min_seq_len, length, lm_main_dir])
