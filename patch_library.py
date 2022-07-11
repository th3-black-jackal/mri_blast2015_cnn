import numpy as np
import random
import os
from glob import glob
import matplotlib
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.feature_extraction.image import extract_patches_2d
import progressbar
import cv2
from skimage.transform import resize
from os.path import exists

progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
np.random.seed()
xrange = range


class PatchLibrary(object):
    def __init__(self, patch_size, train_data, num_samples):
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.train_data = train_data
        self.h = self.patch_size[0]
        self.w = self.patch_size[1]

    def find_patches(self, class_num, num_patches):
        h, w = self.patch_size[0], self.patch_size[1]
        patches, labels = [], np.full(int(num_patches), class_num, 'float')
        print('Finding patches of class {}...'.format(class_num))

        ct = 0
        print('Training data: ', self.train_data)
        while ct < num_patches:
            im_path = random.choice(self.train_data)
            fn = os.path.basename(im_path)
            #If the file name is not founded in the labels path
            if not exists(f'../Labels/{ct}_{fn[-4:]}L.png'):
                continue
            label = io.imread(f'../Labels/{ct}_{fn[-4:]}L.png')
            if len(np.argwhere(label == class_num)) < 10:
                continue
            #Image read can't read from dir, so we will reshape each image independtly
            #img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
            img = self.read_images_from_dir(im_path)
            p = random.choice(np.argwhere(label == class_num))
            p_ix = (p[0] - (h/2), p[0] + ((h+1) / 2), p[1] - (w / 2), p[1] + ((w+1)/2))

            patch = np.array([i[int(p_ix[0]): int(p_ix[1]), int(p_ix[2]):int(p_ix[3])] for i in img])
            if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (h * w):
                continue
            patches.append(patch)
            ct += 1
        return np.array(patches), labels

    def center(self, n, patches):
        sub_patches = []
        for mode in patches:
            subs = np.array([patch[(self.h/2) - (n/2):(self.h / 2) + ((n+1)/2),(self.w/2) - (n/2):(self.w/2)] for patch in mode])
            sub_patches.append(subs)
        return np.array(sub_patches)

    def slice_to_patches(self, filename):
        slices = io.imread(filename).astype('float').reshape(5, 240, 240)[:-1]
        plist = []
        for img in slices:
            if np.max(img) != 0:
                img /= np.max(img)
            p = extract_patches_2d(img, (self.h, self.w))
            plist.append(p)
        print('Patches: ', plist)
        return np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))

    def patches_by_entropy(self, num_patches):
        patches, labels = [], []
        ct = 0
        while ct < num_patches:
            im_path = random.choice(self.train_data)
            fn = os.path.basename(im_path)
            label = io.imread('Labels/' + fn[:-4] + 'L.png')
            if len(np.unique(label)) == 1:
                continue
            img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
            l_ent = entropy(label, disk(self.h))
            top_ent = np.percentile(l_ent, 90)
            if top_ent == 0:
                continue
            highest = np.argwhere(l_ent >= top_ent)
            p_s = random.sample(highest, 3)
            for p in p_s:
                p_ix = (p[0] - (self.h/2), p[0] + ((self.h+1) / 2), p[1] + ((self.w+1) / 2))
                patch = np.array([i[p_ix[0]: p_ix[1], p_ix[2]:p_ix[3]] for i in img])
                if np.shape(patch) != (4, 65, 65):
                    continue
                patches.append(patch)
                labels.append(label[p[0], p[1]])
            ct += 1
            return np.array(patches[: self.num_samples]), np.array(labels[:self.num_samples])

    def read_images_from_dir(self, dir_path):
        images = glob(f'{dir_path}/*')
        return [resize(io.imread(image), (240, 240)) for image in images]


    def make_training_patches(self, entropy=False, balanced_classes=True, classes=None):
        if classes is None:
            classes = [0, 1, 2, 3, 4]
        if balanced_classes:
            per_class = self.num_samples / len(classes)
            patches, labels = [], []
            progress.currval = 0
            for i in progress(xrange(len(classes))):
                p, l = self.find_patches(classes[i], per_class)
                for img_ix in xrange(len(p)):
                    for slice in xrange(len(p[img_ix])):
                        if np.max(p[img_ix][slice]) != 0:
                            p[img_ix][slice] /= np.max(p[img_ix][slice])
                patches.append(p)
                labels.append(l)
            return np.array(patches).reshape(self.num_samples, 4, self.h, self.w), np.array(labels).reshape((self.num_samples))
        else:
            print('Use balanced classes, randome will not work')
