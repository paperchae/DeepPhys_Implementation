import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageEnhance

'''
C(t) : Current video frame at time t
Input :
    Motion Model:
        Normalized Difference 
        (C(t+1) - C(t)) / (C(t+1) + C(t))
    
    Apppearance Model:
        C(t)
'''

video_data_path = '/media/hdd1/UBFC/'
result_data_path = '/media/hdd1/je/Dataset/UBFC_DeepPhys'
video_data_list = os.listdir(video_data_path)

print(video_data_list)
test_vid = np.ones((1, 36, 36, 6))
print(test_vid)


def DeepPhys_Video_Preprocess(path):
    for data in video_data_list:
        try:
            if not (os.path.isdir(video_data_path + data)):
                os.makedirs(os.path.join(result_data_path + data))

                cap = cv2.VideoCapture(video_data_path + data)

                count = 0

                while True:
                    ret, image = cap.read()

                    if not ret:
                        break

                    cv2.imwrite(result_data_path + data + "/%04d.jpg" % count, image)

                    print('%d.jpg done' % count)
                    count += 1

                cap.release()

        except OSError as e:
            if e.errno != e.EEXIST:
                print("Failed to create directory!!!!!")
                raise


def get_normalized_difference(prev_frame, current_frame):
    return (current_frame - prev_frame) / (current_frame + prev_frame)


def preprocess_Image(prev_frame, current_frame):
    return get_normalized_difference(prev_frame, current_frame), current_frame


def video_normalize(channel):
    channel /= np.std(channel)
    return channel
# def DeepPhys_Preprocess_video(path):
#
#     for file in video_data_list:
#         try:
#             if not (os.path.isdir(video_data_path + file)):
#                 os.makedirs(os.path.join(imagePath + file))
#
#                 cap = cv2.VideoCapture(video_data_path + file)
#
#                 count = 0
#
#                 while True:
#                     ret, image = cap.read()
#
#                     if not ret:
#                         break
#
#                     cv2.imwrite(imagePath + file + "/%04d.jpg" % count, image)
#
#                     print('%d.jpg done' % count)
#                     count += 1
#
#                 cap.release()
#
#         except OSError as e:
#             if e.errno != e.EEXIST:
#                 print("Failed to create directory!!!!!")
#                 raise
