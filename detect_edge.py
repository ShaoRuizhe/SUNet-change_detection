import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import os

save_pic_path='%path_to_dataset%/tiles/edges_uav/'
data_path='%path_to_dataset%/tiles/uav/'
sat_data_path='%path_to_dataset%/tiles/sat/'
sat_save_pic_path='%path_to_dataset%/tiles/edges_sat/'
Canny_apertureSize=3

def detect_building_edge_uav():
    canny_low = 180
    canny_high = 210
    hough_threshold = 64
    hough_minLineLength = 16
    hough_maxLineGap = 3
    image_names=os.listdir(data_path)
    for image_name in tqdm.tqdm(image_names):
        img=cv2.imread(data_path+image_name)
        shape=img.shape[:2]
        img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        edges=CannyThreshold(img_gray,canny_low,canny_high)
        lines=cv2.HoughLinesP(edges,hough_rho,hough_theta,hough_threshold,hough_minLineLength,hough_maxLineGap)
        line_pic = np.zeros(shape, np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_pic, (x1, y1), (x2, y2), 1, thickness=1)
        cv2.imwrite(save_pic_path+image_name+'.jpg',line_pic)


def detect_building_edge_sat():
    canny_low = 100
    canny_high = 170
    hough_threshold = 2
    hough_minLineLength = 4
    hough_maxLineGap = 2
    image_names=os.listdir(sat_data_path)
    for image_name in tqdm.tqdm(image_names):
        img=cv2.imread(sat_data_path+image_name)
        shape=img.shape[:2]
        img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        edges=CannyThreshold(img_gray,canny_low,canny_high)
        lines=cv2.HoughLinesP(edges,hough_rho,hough_theta,hough_threshold,hough_minLineLength,hough_maxLineGap)
        line_pic = np.zeros(shape, np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_pic, (x1, y1), (x2, y2), 1, thickness=1)
        cv2.imwrite(sat_save_pic_path+image_name+'.jpg',line_pic)

if __name__ == '__main__':
    detect_building_edge_uav()
    detect_building_edge_sat()
