# script with individual transformations/distortions for the dataset

from torchvision import transforms
import random
from numpy.random import randint
from PIL import ImageDraw
import numpy as np
from wand.image import Image
import cv2


# function to paste randomly located & sized ellipses in blood colour onto image ussing Pillow ------------------------------------------
class BloodBlobRandom(object):
    def __init__(self, blob_diam_min, blob_diam_max, probability):
        self.d_min = blob_diam_min
        self.d_max = blob_diam_max
        self.p = probability

    def __call__(self, sample):

        pil_im = transforms.ToPILImage()
        tens = transforms.ToTensor()

        # random size in given size range [d_min, d_max]
        diams = randint(self.d_min, self.d_max, 2)
        dx = diams[0]
        dy = diams[1]

        # random position
        x_pos = randint(20, 236-dx)
        y_pos = randint(20, 236-dy)

        # if image not heatmap (3 channels vs. 10 channels)
        if sample.shape[0] == 3:
            # transform images with probability = p
            if random.random() < self.p:
                # transform to PIL-image
                image = pil_im(sample)

                # draw ellipse
                draw = ImageDraw.Draw(image)
                draw.ellipse([(x_pos, y_pos), (x_pos+dx,
                                               y_pos+dy)], fill=(136, 8, 8))
                # transform back to tensor
                sample = tens(image)

        return sample

# barrel distortion transformation function (not used)--------------------------------------------------------------------------------


class RadialDistort(object):
    def __init__(self, distort_factor=0.2, distort_diam=0.1):
        self.distort_fac = distort_factor
        self.distort_diam = distort_diam

    def __call__(self, sample):
        sample_np = sample.numpy()  # transform to numpy array
        tens = transforms.ToTensor()

        if sample.shape[0] == 3:
            # transform to wand image
            wand_img = Image.from_array(sample_np, channel_map='RGB')

            # make barrel distortion
            with wand_img as img:
                img.virtual_pixel = 'edge'
                img.distort('barrel', (0.2, 0.0, 0.0, 1.0))

                # convert to opencv/numpy array format
                img_opencv = np.array(img)
                # cv2.imshow("barrel", img_opencv)

            # transform back to tensor
            sample = tens(img_opencv)

        return sample
