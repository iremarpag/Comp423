

import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from   matplotlib.path import Path
from matplotlib import colors
import numpy as np
from PIL import Image
import cv2
from math import sin, cos
import argparse

from kitti_calibration import *

basedir = 'data/kitti_data' # *nix
left_cam_rgb= 'image_2'
label = 'label_2'
velodyne = 'velodyne'
calib = 'calib'

class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]


def loadKittiFiles(frame='000008'):
    '''
    Load KITTI image (.png), calibration (.txt), velodyne (.bin), and label (.txt),  files
    corresponding to a shot.
    '''
    # load image file
    #fn = basedir + left_cam_rgb + frame + '.png'
    fn = os.path.join(basedir, left_cam_rgb, frame + '.png')
    left_cam = cv2.imread(fn)

    # load velodyne file
    #fn = basedir + velodyne + frame + '.bin'
    fn = os.path.join(basedir, velodyne, frame + '.bin')
    velo = np.fromfile(fn, dtype=np.float32).reshape(-1, 4)

    # load calibration file
    #fn = basedir + calib + frame + '.txt'
    fn = os.path.join(basedir, calib, frame + '.txt')
    calib_data = Calibration(fn)

    # load label file
    # label_filename = basedir + label + frame + '.txt'
    label_filename = os.path.join(basedir, label, frame + '.txt')

    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]

    return left_cam, velo, objects, calib_data


def rotation_y(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                   [0, 1, 0],
                   [-s, 0, c]])


def get_corners3d(l, w, h, heading, centerx, centery, centerz):
    ''' Takes an object and a projection matrix (P) and projects the 3d
          bounding box into the image plane.
          Returns:
              corners_3d: (8,3) array in in rect camera coord.
    '''
    R = rotation_y(heading)
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    print("heading: ", heading)

    #x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    #y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    #z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + centerx
    corners_3d[1, :] = corners_3d[1, :] + centery
    corners_3d[2, :] = corners_3d[2, :] + centerz

    return np.transpose(corners_3d)


def get_corners2d(corners_3d, P):
    ''' Takes a projection matrix (P) and projects the 3d
          bounding box into the image plane.
          Returns:
              corners_2d: (8,2) array in left image coord.
    '''
    corners_2d = project_to_image(corners_3d, P)
    return corners_2d


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
    input: pts_3d: nx3 matrix
           P:      3x4 projection matrix
    output: pts_2d: nx2 matrix

    P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
    => normalize projected_pts_2d(2xn)

    <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
        => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the projected 3d box.
    '''
    qs = qs.astype(np.int32)
    print(" 2d coordinates: ", qs.shape)
    for k in range(0, 2):
      i, j = k, (k + 1) % 4
      # use CV_AA for opencv2
      #print(i, j, k)
      cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

      i, j = k + 4, (k + 1) % 4 + 4
      #print(i, j, k)
      cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

      i, j = k, k + 4
      #print(i, j, k)
      cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image


def show_image_with_3dboxes(img, objects, calibd):
    ''' Show image with 3D bounding boxes '''
    img_new = np.copy(img)
    P2_rect = calibd.P
    print(len(objects))
    for i, obj in enumerate(objects):
        if obj.type == 'DontCare': continue
        if i == 4: # or i == 3:
            print("i ", i, " center: ", obj.t[0], obj.t[1], obj.t[2], obj.ry)
            corners_3d = get_corners3d(obj.l, obj.h, obj.w, obj.ry, obj.t[0], obj.t[1], obj.t[2])
            corners_2d = get_corners2d(corners_3d, P2_rect)
            img_new = draw_projected_box3d(img_new, corners_2d)
    return img_new
    Image.fromarray(img_new).show()


if __name__ == '__main__':

    left_cam, velo, objects, calib_data = loadKittiFiles()
    # print("camera shape: ", left_cam.size)
    # print("velo: ", velo.shape)
    # print("label data: ", len(objects))
    print("calib data: ", calib_data)

    #cv2.imshow('Image', left_cam)
    #cv2.waitKey(0)

    img_boxes = show_image_with_3dboxes(left_cam, objects, calib_data)
    cv2.imshow('3D boxes', img_boxes)
    cv2.waitKey(0)
