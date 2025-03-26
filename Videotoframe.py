# -*- coding: utf-8 -*-

import cv2
import os
def extract_frames(video_path, output_folder):
  """Extracts frames from a video and saves them into a single folder.

  Args:
      video_path (str): Path to the video file.
      output_folder (str): Path to the output folder.
  """

  vidcap = cv2.VideoCapture(video_path)
  success, image = vidcap.read()
  count = 0

  while success:
      cv2.imwrite(os.path.join(output_folder, f"frame_{count}.jpg"), image)  # Save frame as JPEG
      success, image = vidcap.read()
      count += 1

  vidcap.release()
  print(f"Extracted frames from {video_path} into {output_folder}")

# Specify the path to the video folder and output folder
video_folder  = "Video_Dataset/Theft"
output_folder  = "Frame_Dataset/Theft"

# Ensure output folder exists
if not os.path.exists(output_folder):
  os.makedirs(output_folder)

for filename in os.listdir(video_folder):
  if filename.endswith((".mp4", ".avi", ".mov")):  # Check for supported video extensions
    video_path = os.path.join(video_folder, filename)
    extract_frames(video_path, output_folder)