
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

from kitti_boxes import *
from kitti_calibration import *

basedir = 'computer_vision/kitti_3dobj_det_chk/data' # *nix
left_cam_rgb= 'image_2'
label = 'label_2'
velodyne = 'velodyne'
calib = 'calib'


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def write_pointcloud(filename,xyz_points,rgb_points=None):
    import struct

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()


def in_hull(p, hull):
    ''' TODO: implement this part '''
    from scipy.spatial import Delaunay
    hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_object(pc, bounding_box_3d):
    ''' pc: (N,3), box3d: (8,3) '''
    ''' TODO: implement this part '''
    box3d_roi_inds = in_hull(pc[:,0:3], bounding_box_3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


if __name__ == '__main__':

    img, pc_velo, objects, calib_data = loadKittiFiles()
    # print("camera shape: ", left_cam.size)
    # print("velo: ", velo.shape)
    # print("label data: ", len(objects))
    # print("calib data: ", calib_data)

    print("pc velo: ", pc_velo.shape)
    #points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    #np.savetxt('_test_file_Lidar.txt', pc_velo)
    #write_pointcloud('test_file_Lidar.ply', pc_velo[:,:3], rgb_points=None)

    basedir = 'computer_vision/kitti_3dobj_det_chk/data'  # *nix
    left_cam_rgb = 'image_2'
    label = 'label_2'
    velodyne = 'velodyne'
    calib = 'calib'
    calib_filename = os.path.join(basedir, "calib", '000008.txt')
    calib = Calibration(calib_filename)

    pc_rect = np.zeros_like(pc_velo)
    pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
    pc_rect[:, 3] = pc_velo[:, 3]
    img_height, img_width, img_channel = img.shape
    fov_velo, pc_image_coord, img_fov_inds = get_lidar_in_image_fov( \
        pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)

    #print("pc_rect: ", pc_rect.shape)
    print("pc velo: ", pc_velo.shape)
    print("fov velo: ", fov_velo.shape)
    print(" pc image coordinates: ", pc_image_coord.shape)
    print("img_fov inds: ", img_fov_inds.shape)

    #write_pointcloud('lidar_velodyne_fov.ply', fov_velo[:, :3], rgb_points=None)

    pc_image_coord1 = pc_image_coord[img_fov_inds, :]
    print(" pc image coordinates after: ", pc_image_coord.shape)
    np.save('img_coord_pc', pc_image_coord)

    from kitti_boxes import get_corners3d

    for obj_idx in range(len(objects)):
        # 2D BOX: Get pts rect backprojected
        box2d = objects[obj_idx].box2d
        xmin, ymin, xmax, ymax = box2d
        box_fov_inds = (pc_image_coord1[:, 0] < xmax) & \
                       (pc_image_coord1[:, 0] >= xmin) & \
                       (pc_image_coord1[:, 1] < ymax) & \
                       (pc_image_coord1[:, 1] >= ymin)
        # box_fov_inds = box_fov_inds & img_fov_inds
        #pc_in_box_fov = pc_rect[box_fov_inds, :]
        pc_in_box_fov = fov_velo[box_fov_inds, :]
        print("pc in bb velo: ", pc_in_box_fov.shape)

        box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                       (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & \
                       (pc_image_coord[:, 1] >= ymin)

        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds, :]
        #pc_in_box_fov = fov_velo[box_fov_inds, :]
        print("pc in bb rect: ", pc_in_box_fov.shape)

        ply_file = "objrect_" + str(obj_idx) + ".ply"
        write_pointcloud(ply_file, pc_in_box_fov[:,:3])


        obj = objects[obj_idx]
        corners_3d = get_corners3d(obj.l, obj.h, obj.w, obj.ry, obj.t[0], obj.t[1], obj.t[2])
        print("corners shape: ", corners_3d.shape)
        corners_3d_velo = calib.project_rect_to_velo(corners_3d)
        #print("corners_3d_velo shape: ", corners_3d_velo.shape)
        #gt_obj_pcd, roi_inds = extract_object(pc_velo, corners_3d_velo)
        #print("gt pc in bb: ", gt_obj_pcd.shape, roi_inds.sum())
        gt_obj_pcd, roi_inds = extract_object(pc_rect, corners_3d)
        print("gt pc in bb: ", gt_obj_pcd.shape, roi_inds.sum())

        ply_file = "objgt_" + str(obj_idx) + ".ply"
        write_pointcloud(ply_file, gt_obj_pcd[:,:3])

        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                        box2d_center_rect[0, 0])

        print("frustum angle: ", frustum_angle)
