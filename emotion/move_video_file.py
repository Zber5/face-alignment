import shutil
import os

source_dir = 'C:/Users/Zber/Desktop/SavedData_MIMO'
target_dir = 'C:/Users/Zber/Desktop/Emotion_video'

folder_names = os.listdir(source_dir)


for folder_name in folder_names:
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            shutil.copy(os.path.join(folder_path, file_name), target_dir)

