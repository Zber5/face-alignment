import cv2
import numpy as np
import time
import os


def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # read the video
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, old_frame = cap.read()
    # old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame



def lucas_kanade_method(video_path, output_as_video=True):
    camera_fps = 30
    width = 1080
    height = 720
    if output_as_video:
        # video_output = os.path.join(output_path,
        #                             "{}_landmark_flm.avi".format(os.path.basename(os.path.normpath(output_path))))
        video_output = os.path.join("C:\\Users\\Zber\\Desktop", "optical_flow.avi")
        # fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        video_writer = cv2.VideoWriter(str(video_output),
                                       fourcc,
                                       camera_fps,
                                       (width, height))


    cap = cv2.VideoCapture(video_path)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # b_img = np.zeros(old_frame.shape)
    #
    # pTime = 0
    # if True:
    #     cTime = time.time()
    #     fps = 1 / (cTime - pTime)
    #     pTime = cTime
    #     cv2.putText(black_img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    #     WindowName = "Image"
    #     cv2.imshow("Image", black_img)
    #     cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
    #     cv2.waitKey(10)


    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)
        if output_as_video:
            video_writer.write(img)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        if k == ord("c"):
            mask = np.zeros_like(old_frame)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    if output_as_video:
        video_writer.release()




from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=["farneback", "lucaskanade", "lucaskanade_dense", "rlof"],
        required=False,
        default="farneback",
        help="Optical flow algorithm to use",
    )
    parser.add_argument(
        "--video_path", default="videos/cat.mp4", help="Path to the video",
    )

    args = parser.parse_args()
    args.algorithm = "lucaskanade"
    # video_path = args.video_path
    video_path = "C:/Users/Zber/Desktop/SavedData_MIMO/Joy_0/Joy_0_landmark_flm.avi"
    # video_path = "C:/Users/Zber/Desktop/SavedData/Joy_0/Joy_0_landmark_flm.avi"
    if args.algorithm == "lucaskanade":
        lucas_kanade_method(video_path)
    elif args.algorithm == "lucaskanade_dense":
        method = cv2.optflow.calcOpticalFlowSparseToDense
        dense_optical_flow(method, video_path, to_gray=True)
    elif args.algorithm == "farneback":
        method = cv2.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback's algorithm parameters
        dense_optical_flow(method, video_path, params, to_gray=True)
    elif args.algorithm == "rlof":
        method = cv2.optflow.calcOpticalFlowDenseRLOF
        dense_optical_flow(method, video_path)


if __name__ == "__main__":
    main()