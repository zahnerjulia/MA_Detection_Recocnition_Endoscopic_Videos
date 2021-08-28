import torch
import config as cfg
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from preproc.load_data import Dataset_pyt
import glob
from skimage import color
import cv2
import os
from natsort import natsorted
import imageio

# quick visualization of predictions and heatmaps


def visualize():

    for batch in range(0, 6):

        # load predictions and heatmaps (labels)
        pred = np.load(cfg.save_path_pred + cfg.pred_name +
                       'batch_{}.npy'.format(batch))
        label = np.load(cfg.save_path_hmp + 'batch_{}.npy'.format(batch))

        fig = plt.figure()
        plt.title('test_batch{}'.format(batch))
        plt.ylabel('Predictions & Corresponding Heatmaps')
        plt.xlabel('Labels')

        for i in range(0, np.shape(pred)[0], 5):
            print(i)
            for j in range(0, np.shape(pred)[1]):
                # show all predicted heatmaps for every fifth frame in the batch
                ax = fig.add_subplot(
                    (np.shape(pred)[0]//5+1) * 2, np.shape(pred)[1], i//5*20 + j+1)
                ax.set_axis_off()
                ax1.set_ylabel('frame {}'.format(i))
                plt.imshow(pred[i, j, :, :])
                # show all ground truth heatmaps for every fifth frame in the batch
                ax1 = fig.add_subplot(
                    (np.shape(pred)[0]//5+1) * 2, np.shape(pred)[1], i//5*20 + 10 + j+1)
                ax1.set_axis_off()
                ax1.set_ylabel('frame {}'.format(i))
                plt.imshow(label[i, j, :, :])

        # Save figure
        if os.path.exists(cfg.save_path_visual + cfg.pred_name) == False:
            os.mkdir(cfg.save_path_visual + cfg.pred_name)
        plt.savefig(cfg.save_path_visual + cfg.pred_name +
                    'test_batch{}.png'.format(batch))
        plt.show()

# Make overlays with image predication and ground truth


def visualize_2():

    # load images for overlay
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(cfg.image_cropped_size)
    ])

    image_dataset = Dataset_pyt(cfg.base_data_path,
                                cfg.test_data_path, transform=data_transform)
    image_data = DataLoader(image_dataset, num_workers=4)

    for batch, (img, hmp, bl) in enumerate(image_data):
        if batch == 0:
            continue
        if os.path.exists(cfg.gif_path) == False:
            os.mkdir(cfg.gif_path)
        if os.path.exists(cfg.gif_path + 'batch_{}/'.format(batch)) == False:
            os.mkdir(cfg.gif_path + 'batch_{}/'.format(batch))

        # load image for overlay
        image = torch.squeeze(img).numpy()
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 2, 3)

        # load predictions for overlay
        pred = np.load(cfg.save_path_pred + cfg.pred_name +
                       'batch_{}.npy'.format(batch))

        # load label for overlay
        label = np.load(cfg.save_path_hmp + 'batch_{}.npy'.format(batch))

        # make overlay (loop through frames and labels)
        for i in range(0, np.shape(pred)[0]):
            for j in range(0, np.shape(pred)[1]):

                # find index of label center
                idx = np.where(label[i, j, :, :] == 1)

                # convert image to grayscale
                greyscale_image = color.rgb2gray(
                    image[i, :, :, :])

                # inizialize white rgb image for label layer
                x = np.ones((256, 256, 3), np.uint8)*255

                # if a label is present, draw blue circle onto label layer at index point
                if idx[0].size > 0:
                    x = cv2.circle(
                        x, (idx[1][0], idx[0][0]), radius=1, color=(0, 0, 255), thickness=4)

                # mask image that one can only see the blue label point in the overlay
                alpha = ~np.all(x == 255, axis=2) * 255
                rgba = np.dstack((x, alpha)).astype(np.uint8)

                # grayscale image layer
                plt.imshow(greyscale_image[:, :], cmap='gray')

                # prediction layer
                plt.imshow(pred[i, j, :, :], alpha=pred[i, j, :, :])

                # label point layer
                plt.imshow(rgba)

                # save frames for all the labels and batches
                plt.savefig(
                    cfg.gif_path + 'batch_{}/'.format(batch) + 'frame_{}_label_{}.png'.format(i, j))


# make gif out of overlay for each batch and seve them
def make_gif():
    images = []
    for i in range(0, 2):
        for j in range(0, 10):
            frames = natsorted(glob.glob(
                cfg.gif_path+'batch_{}/frame_*_label_{}.png'.format(i, j)))
            for frame in frames:
                print(frame)
                images.append(imageio.imread(frame))
                imageio.mimsave(
                    cfg.gif_path + 'batch_{}/label_{}.gif'.format(i, j), images, format='GIF', fps=2)


if __name__ == "__main__":
    visualize()
    visualize_2()
    make_gif()
