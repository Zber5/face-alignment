import subprocess
import os
import threading
import cv2
import time

from multiprocessing import Pool

VIDEO_EXTENSIONS = ['mp4', 'webm', 'avi']


def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video../n")
    video_name = os.path.basename(input_loc).split(".")[0]
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/{}_{:0>5}.jpg".format(video_name,count), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames./n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break


def makefile(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


if __name__=="__main__":
    video_dir = "C:/Users/Zber/Desktop/Emotion_video"
    frame_dir = "C:/Users/Zber/Desktop/Emotion_frame"

    pool = Pool()
    # res = pool.apply_async(f, (20,))
    # pool.map()

    videos = []
    frames = []

    for video_file in os.listdir(video_dir):
        if is_video_file(video_file):
            video_name = os.path.join(video_dir, video_file)
            frame_output_path = os.path.splitext(video_name.replace(video_dir, frame_dir))[0]
            videos.append(video_name)
            frames.append(frame_output_path)
            # makefile(frame_output_path)
            # pool.apply_async(video_to_frames, (video_name, frame_output_path,))

            # video_to_frames(video_name, frame_output_path)

    pool.starmap(video_to_frames, zip(videos, frames))




