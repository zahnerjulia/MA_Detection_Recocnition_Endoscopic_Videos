# script to create csv-file with image paths and coordinates of labels for easy loading of the data
import json
import os
import glob
import random
from collections import defaultdict
from natsort import natsorted

# base directory where image data is stored
base_dir = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/SerraVideos/'

# 26 patients (#10 has no data)
num_of_patients = 26

# -------------------------------------------------------------------------------------------------------------------------------------


def create_csv_annot(base_dir, num_of_patients):

    # check annotation files and write files with tags of lenghth != 1 to bad files
    all_files = glob.glob(base_dir+'Annotations/*/*-asset.json')
    badfiles = set()
    all_tags = defaultdict(int)

    for f in all_files:
        with open(f) as fh:
            ob = json.load(fh)
            for region in ob['regions']:
                if not len(region['tags']) == 1:
                    badfiles.add(f)
                else:
                    all_tags[region['tags'][0].lower()] += 1

    # group annotations into labels based on name
    group_annots = {
        0: 'coana',


        1: 'floor',


        2: ['inferior t', 'inferior turbinate', 'it'],


        3: ['middle t', 'middle turbinate', 'mt'],


        4: 'ostium',


        5: ['recess', 'sfen etm recess'],


        6: ['septum', 'seputm'],


        7: 'st',


        8: 'uncinate',


        9: 'bulla'
    }

    tags_used = list(range(10))  # select which labels should be used
    patient_num = list(range(num_of_patients))
    del(patient_num[10])  # patient 10 is empty
    patients = ['patient_{:04d}'.format(i) for i in patient_num]
    images = []  # inizialize list for all images

    # make dirs to save lists ---------------------------------------------------------------------------------------------------------------
    save_path_list_all = '/home/zahnerj/MA_endo_julia/preproc/data_lists/setup'
    save_path_list_split = '/home/zahnerj/MA_endo_julia/preproc/data_lists/split/shuffle'
    save_path_list_split_sorted = '/home/zahnerj/MA_endo_julia/preproc/data_lists/split/sorted/'

    os.makedirs(save_path_list_all, exist_ok=True)
    os.makedirs(save_path_list_split, exist_ok=True)
    os.makedirs(save_path_list_split_sorted, exist_ok=True)

    # function to extract the data from the jsons and write the csv-file --------------------------------------------------------------------
    with open(os.path.join(save_path_list_all, 'all.csv'), 'w') as g:
        for patient in patients:

            # directory where json data is
            jsondir = os.path.join(base_dir, 'Annotations', patient)

            # all jsons for one patient
            jsons = glob.glob(jsondir+'/*-asset.json')

            # extract side
            side_path = jsondir + '/side.txt'
            with open(side_path) as f:
                side = f.read().strip()

            # loop through json files
            for filename in sorted(jsons):
                line = ''
                if filename in badfiles:
                    continue
                with open(filename) as f:
                    annotation_file = json.load(f)
                    # extract timestamp
                    try:
                        t = annotation_file['asset']['timestamp']
                        t = str(t).replace('.', '-')
                    except KeyError:
                        continue

                    # define image path to write into csv-file
                    image = os.path.join(
                        '/scratch_net/biwidl214/zahnerj/data/flipped_resized_images', patient, patient+'#t={}.png'.format(t))
                    assert os.path.isfile(image), image + ' does not exist'
                    line += image  # add image path to line to write later

                    # extract label point from json
                    num_tags_found = 0
                    for tag in tags_used:
                        tag_annotated = False
                        for region in annotation_file['regions']:
                            if len(region['tags']) > 1:
                                continue
                            else:
                                if region['tags'][0].lower() in group_annots[tag]:
                                    tag_annotated = True
                                    assert len(region['points']) == 4
                                    x = 0
                                    y = 0

                                    # take average of the four (x,y) locations of the bounding box
                                    for point in region['points']:
                                        x += point['x'] / 4
                                        y += point['y'] / 4

                                    # mirror points based on side
                                    if side == 'right':
                                        x = 1920-x
                                        assert x > 0
                                        assert y > 0
                                    else:
                                        assert side == 'left'

                                    # scale label points to get label points in the 256 * 256 pixel image
                                    x *= 256/1920
                                    y *= 256/1080

                                    num_tags_found += 1

                                    # add label point (x,y) to line
                                    line += ', {}, {}'.format(int(x), int(y))
                                    break
                        if not tag_annotated:
                            # add NaN if no label there
                            line += ', {}, {}'.format('NaN', 'NaN')

                g.write(line + '\n')
                images.append(image)

    random.shuffle(images)  # shuffle list of images (not used for this work)

    # make lists of filepaths for train and test datasets (take only files that have correct annotations) ------------------------------
    train_patient_numbers = list(range(20))
    del(train_patient_numbers[10])  # patient_0010 is empty
    train_patients = ['patient_{:04d}'.format(
        i) for i in train_patient_numbers]

    test_patient_numbers = list(range(20, 26))
    test_patients = ['patient_{:04d}'.format(i) for i in test_patient_numbers]

    with open(os.path.join(save_path_list_split, 'train.txt'), 'w') as g:
        for image in images:
            if any([train in image for train in train_patients]):
                g.write(image + '\n')

    with open(os.path.join(save_path_list_split_sorted, 'train.txt'), 'w') as g:
        for image in natsorted(images):
            if any([train in image for train in train_patients]):
                g.write(image + '\n')

    with open(os.path.join(save_path_list_split, 'train10.txt'), 'w') as g:
        for image in images:
            if any([train in image for train in train_patients[:1]]):
                g.write(image + '\n')

    with open(os.path.join(save_path_list_split_sorted, 'train10.txt'), 'w') as g:
        for image in natsorted(images):
            if any([train in image for train in train_patients[:1]]):
                g.write(image + '\n')

    with open(os.path.join(save_path_list_split, 'test.txt'), 'w') as g:
        for image in images:
            if any([test in image for test in test_patients]):
                g.write(image + '\n')

    with open(os.path.join(save_path_list_split_sorted, 'test.txt'), 'w') as g:
        for image in natsorted(images):
            if any([test in image for test in test_patients]):
                g.write(image + '\n')
