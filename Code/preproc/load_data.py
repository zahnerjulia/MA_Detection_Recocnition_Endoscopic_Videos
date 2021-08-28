# file with everything used to preprocess, load the data and make the heatmaps

import json
import math
import cv2
import albumentations
from albumentations.pytorch import ToTensorV2
from preproc.distortion import RadialDistort, BloodBlobRandom
import torch
import pandas as pd
import os
import numpy as np
import re
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.transforms.transforms import CenterCrop, Resize, ToPILImage
import glob
from natsort import natsorted
import config as cfg


# Landmark class ------------------------------------------------------------------------------------------------------------------------
class Landmark(object):
    def __init__(self,
                 coords=None,
                 is_valid=None,
                 scale=1.0,
                 value=None):
        self.coords = coords
        self.is_valid = is_valid
        if self.is_valid == None:
            self.is_valid = self.coords is not None
        self.scale = scale
        self.value = value


# Heatmap class -----------------------------------------------------------------------------------------------------------------------
class Heatmap(object):
    def __init__(self, label_id, sigma, is_valid=None):
        self.label_id = label_id
        self.sigma = sigma
        self.is_valid = is_valid
        if self.is_valid == None:
            self.is_valid = True

# Heatmap Image Generator Class --------------------------------------------------------------------------------------------------------


class HeatmapImageGenerator(object):
    """
    Generates numpy arrays of Gaussian landmark images for the given parameters.
    :param image_size: Output image size
    :param sigma: Sigma of Gaussian
    :param scale_factor: Every value of the landmark is multiplied with this value
    :param normalize_center: if true, the value on the center is set to scale_factor
                             otherwise, the default gaussian normalization factor is used
    :param size_sigma_factor: the region size for which values are being calculated
    """

    def __init__(self,
                 image_size,
                 sigma,
                 scale_factor,
                 normalize_center=True,
                 size_sigma_factor=10):
        self.image_size = image_size
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.dim = len(image_size)
        self.normalize_center = normalize_center
        self.size_sigma_factor = size_sigma_factor

 # -------------------------------------------------------------------------------------------------------------------------------------

    def generate_heatmap(self, coords, sigma_scale_factor):
        """
        Generates a numpy array of the landmark image for the specified point and parameters.
        :param coords: numpy coordinates ([x], [x, y] or [x, y, z]) of the point.
        :param sigma_scale_factor: Every value of the gaussian is multiplied by this value.
        :return: numpy array of the landmark image.
        """
        # landmark holds the image
        heatmap = np.zeros(self.image_size, dtype=np.float32)

        # flip point from [x, y, z] to [z, y, x]
        flipped_coords = np.flip(coords, 0)
        region_start = (flipped_coords - self.sigma *
                        self.size_sigma_factor / 2).astype(int)
        region_end = (flipped_coords + self.sigma *
                      self.size_sigma_factor / 2).astype(int)

        region_start = np.maximum(0, region_start).astype(int)
        region_end = np.minimum(self.image_size, region_end).astype(int)

        # return zero landmark, if region is invalid, i.e., landmark is outside of image
        if np.any(region_start >= region_end):
            return heatmap

        region_size = (region_end - region_start).astype(int)

        sigma = self.sigma * sigma_scale_factor
        scale = self.scale_factor

        if not self.normalize_center:
            scale /= math.pow(math.sqrt(2 * math.pi) * sigma, self.dim)

        if self.dim == 1:
            dx = np.meshgrid(range(region_size[0]))
            x_diff = dx + region_start[0] - flipped_coords[0]

            squared_distances = x_diff * x_diff

            cropped_heatmap = scale * \
                np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

            heatmap[region_start[0]:region_end[0]] = cropped_heatmap[:]

        if self.dim == 2:
            dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
            x_diff = dx + region_start[0] - flipped_coords[0]
            y_diff = dy + region_start[1] - flipped_coords[1]

            squared_distances = x_diff * x_diff + y_diff * y_diff

            cropped_heatmap = scale * \
                np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

            heatmap[region_start[0]:region_end[0],
                    region_start[1]:region_end[1]] = cropped_heatmap[:, :]

        elif self.dim == 3:
            dy, dx, dz = np.meshgrid(range(region_size[1]), range(
                region_size[0]), range(region_size[2]))
            x_diff = dx + region_start[0] - flipped_coords[0]
            y_diff = dy + region_start[1] - flipped_coords[1]
            z_diff = dz + region_start[2] - flipped_coords[2]

            squared_distances = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff

            cropped_heatmap = scale * \
                np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

            heatmap[region_start[0]:region_end[0],
                    region_start[1]:region_end[1],
                    region_start[2]:region_end[2]] = cropped_heatmap[:, :, :]

        return heatmap

# -------------------------------------------------------------------------------------------------------------------------------------

    def generate_heatmaps(self, landmarks, stack_axis):
        """
        Generates a numpy array landmark images for the specified points and parameters.
        :param landmarks: List of points. A point is a dictionary with the following entries:
            'is_valid': bool, determines whether the coordinate is valid or not
            'coords': numpy coordinates ([x], [x, y] or [x, y, z]) of the point.
            'scale': scale factor of the point.
        :param stack_axis: The axis where to stack the np arrays.
        :return: numpy array of the landmark images.
        """
        heatmap_list = []

        for landmark in landmarks:
            if landmark.is_valid:
                heatmap_list.append(self.generate_heatmap(
                    landmark.coords, landmark.scale))
            else:
                heatmap_list.append(np.zeros(self.image_size, np.float32))

        heatmaps = np.stack(heatmap_list, axis=stack_axis)

        return heatmaps

# -------------------------------------------------------------------------------------------------------------------------------------


def load_csv(file_name, num_landmarks, dim):
    """
    Generates landmark objects from a csv file storing the coordinates as numbers and NaNs if not available
    :param 'file_name': path to csv-file with landmark coordinate information
    :param 'num_landmarks': number of landmarks
    :param 'dim': dimension number (2D or 3D coordinates)
    :return: Dictonary with landmark-objects storing the landmark coordinates as numpy float arrays, id's are the paths to the associated images
    """
    landmarks_dict = {}
    counter = np.zeros(num_landmarks)
    counter_dict = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id = row[0]
            landmarks = []
            count = []
            num_entries = dim * num_landmarks + 1
            assert num_entries == len(
                row), 'number of row entries ({}) and landmark coordinates ({}) do not match'.format(num_entries, len(row))
            # print(len(points_dict), name)
            for i in range(1, dim * num_landmarks + 1, dim):
                # print(i)
                if np.isnan(float(row[i])):
                    landmark = Landmark(None, False)
                    count.append(0)
                else:
                    count.append(1)
                    idx = int((i-1)/dim)
                    counter[idx] += int(1)
                    if dim == 2:
                        coords = np.array(
                            [float(row[i]), float(row[i + 1])], np.float32)
                    elif dim == 3:
                        coords = np.array([float(row[i]), float(
                            row[i + 1]), float(row[i + 2])], np.float32)
                    landmark = Landmark(coords)
                landmarks.append(landmark)
            counter_dict[id] = count
            landmarks_dict[id] = landmarks

    num_patients = 26
    count_per_patient = np.zeros((num_patients, num_landmarks))
    for key in counter_dict:
        for i in range(num_patients):
            if 'patient_{:04d}'.format(i) in key:
                count_per_patient[i][:] += counter_dict[key]

    return landmarks_dict

# Labeled dataset class ----------------------------------------------------------------------------------------------------------------


class Dataset_pyt(Dataset):
    def __init__(self, base_data_all, data_path, heatmap_size=[256, 256], num_of_landmarks=10, dim=2, sigma=30, size_sigma_factor=10, scale_factor=1, normalize_center=True, transform=None):
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.normalize_center = normalize_center
        self.size_sigma_factor = size_sigma_factor
        self.base_data_all = base_data_all
        self.data_path = data_path
        self.num_of_landmarks = num_of_landmarks
        self.dim = dim
        self.transform = transform

        # load all label points, make landmarks and retrun a dictonary of landmarks
        self.landmarks_dict = load_csv(
            self.base_data_all, self.num_of_landmarks, self.dim)

        # initialze dictionary to stor heatmaps later
        self.heatmaps_dict = {}

        # one batch per patient (indices where new patient starts)
        self.data_file = pd.read_csv(self.data_path, header=None)
        if 'test' in self.data_path:
            self.batch_indices = [0, 43, 90, 143, 188, 234, 274]
        else:
            self.batch_indices = [0, 55, 110, 142, 189, 286, 341, 424, 495,
                                  604, 697, 751, 812, 865, 927, 977, 1039, 1092, 1145, 1201]

        for i in range(len(self.data_file)):

            # load data path as id
            id = self.data_file.iloc[i, 0]

            # generate heatmap for given landmark with the HeatmapImageGenerator class
            heatmap_generator = HeatmapImageGenerator(
                self.heatmap_size, self.sigma, self.scale_factor, self.normalize_center, self.size_sigma_factor)

            # write heatmaps to dictionary
            self.heatmaps_dict[id] = heatmap_generator.generate_heatmaps(
                self.landmarks_dict[id], 2)

    # return number of batches
    def __len__(self):
        return len(self.batch_indices)-1

    # get function to access the files from the dataloader
    def __getitem__(self, idx):
        start_idx = self.batch_indices[idx]
        stop_idx = self.batch_indices[idx+1]
        batch_len = stop_idx-start_idx
        images = torch.empty(size=(batch_len, 3, 256, 256))
        heatmaps = torch.empty(size=(batch_len, 10, 256, 256))

        for i in range(start_idx, stop_idx):
            path_id = self.data_file.iloc[i, 0]
            image = io.imread(path_id)
            heatmap = self.heatmaps_dict[path_id]

            # preprocess images and heatmaps
            if self.transform:
                image = self.transform(image)
                heatmap = self.transform(heatmap)

            # write to tensor
            images[i-start_idx] = image
            heatmaps[i-start_idx] = heatmap

        return images, heatmaps, batch_len

# --------------------------------------------------------------------------------------------------------------------------------------


def Vid_to_frames(path, num_of_patients, start_frames_video):
    """
    Extracts images from video data at 10 frames per second, saves them, create txt-file with paths to the images 
    :param 'path': path to video data
    :param 'num_of_patients': number of patients (26)
    :param 'start_frames_video': frame indices where video contains data from inside the body
    :return: nothing ()
    """
    all_files = glob.glob(path+'/*/video.mp4')
    done = False
    start_frame = start_frames_video

    for j, f in enumerate(natsorted(all_files)):
        path_name = os.path.split(f)
        splitted = os.path.split(path_name[0])
        cap = cv2.VideoCapture(f)

        # make directories
        if os.path.exists(cfg.frames_path+splitted[1]+'/') == False:
            os.makedirs(cfg.frames_path+splitted[1]+'/')

        # check if frames where already extracted
        if os.path.exists(cfg.frames_path+splitted[1]+'/done.txt'):
            with open(cfg.frames_path+splitted[1]+'/done.txt', 'r') as fp:
                done = json.load(fp)

        if done:
            continue

        i = 0

        # extract frames
        while(cap.isOpened()):
            success, image = cap.read()
            if success == False:
                break
            if i//3 >= start_frame[j]:
                if not os.path.exists(cfg.frames_path+splitted[1]+'/frame_' + str(i//3)+'.jpg') and i % 3 == 0:
                    cv2.imwrite(cfg.frames_path+splitted[1]+'/frame_' +
                                str(int(i//3))+'.jpg', image)
            else:
                continue
            i += 1

        cap.release()
        cv2.destroyAllWindows()

        with open(cfg.frames_path+splitted[1]+'/done.txt', 'w+') as fp:
            json.dump(True, fp, indent=2)

    all_frames = []
    all_frames_new = []
    all_frames_splitted_50 = []

    for i in range(num_of_patients):
        all_frames.append(
            natsorted(glob.glob(cfg.frames_path+'patient_*'+str(i)+'/frame_*.jpg')))

    for i, a in enumerate(all_frames):
        all_frames_new.append(a[start_frame[i]:])

    # create batches of 50 images
    for a in all_frames_new:
        for i in range(0, len(a), 50):
            if len(a[i:i+50]) < 10:
                continue
            all_frames_splitted_50.append(a[i:i+50])

    # save txt-file
    with open(cfg.data_list_path+'all_frames.txt', 'w') as fp:
        json.dump(all_frames_splitted_50, fp, indent=2)

# Video Dataset used for pretraining ----------------------------------------------------------------------------------------------------


class Dataset_vid(Dataset):
    def __init__(self, path_data, transform=None):
        # self.path = path
        self.transform = transform

        with open(path_data + "all_frames.txt", "r") as fp:
            self.frames_list = json.load(fp)
        # print(self.frames_list)
        # print(self.frames_list)
        # print(self.frames_list[0][1])
        with open(path_data+"all_fps.txt", "r") as fp:
            self.fps = json.load(fp)
        # print(self.fps)

        # self.frames_list, self.fps = Vid_to_frames(self.path, num_of_pat)

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        batch_len = len(self.frames_list[idx])
        images = torch.empty(size=(batch_len, 3, 256, 256))

        for i in range(batch_len):
            path_id = self.frames_list[idx][i]
            img = io.imread(path_id)

            if self.transform:
                img = self.transform(img)

            images[i] = img
        return images, batch_len
