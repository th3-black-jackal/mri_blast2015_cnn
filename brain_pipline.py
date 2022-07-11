import numpy as np
import subprocess
import random
import progressbar
from glob import glob
from skimage import io

np.random.seed(5)
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])


class BrainPipeline(object):
    def __init__(self, path, n4itk = True, n4itk_apply = False):
        self.path = path
        self.n4itk = n4itk
        self.n4itk_apply = n4itk_apply
        self.modes = ['flair', 't1', 't1c', 't2', 'gt']
        self.slices_by_mode, n = self.read_scans()
        self.slices_by_slice = n
        self.normed_slices = self.norm_slices()

    def read_scans(self):
        slices_by_mode = np.zeros((5, 155, 240, 240))
        slices_by_slice = np.zeros((155, 5, 240, 240))
        flair = glob(self.path + '/*Flair*.mha')
        t2 = glob(self.path + '/*_T2*.mha')
        gt = glob(self.path + '/*more*.mha')
        t1s = glob(self.path + '/*T1*.mha')
        t1_n4 = glob(self.path + '/*T1/*_n.mha')
        t1 = [scan for scan in t1s if scan not in t1_n4]

        scans = [flair[0], t1[0], t1[1], t2[0], gt[0]]
        #print('Scans: ', scans)
        if self.n4itk_apply:
            #print('Applying bias correction')
            for t1_path in t1:
                self.n4itk_norm(t1_path)
            #print('T1_N4: ', t1_n4)
            scans = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
        elif self.n4itk:
            scans = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]

        for scan_idx in range(5):
            slices_by_mode[scan_idx] = io.imread(scans[scan_idx], plugin='simpleitk').astype(float)
        for mode_ix in range(slices_by_mode.shape[0]):
            for slice_ix in range(slices_by_mode.shape[1]):
                slices_by_slice[slice_ix][mode_ix] = slices_by_mode[mode_ix][slice_ix]
        return slices_by_mode, slices_by_slice

    def norm_slices(self):
        print('Normalizing slices...')
        normed_slices = np.zeros((155, 5, 240, 240))
        for slice_ix in range(155):
            normed_slices[slice_ix][-1] = self.slices_by_slice[slice_ix][-1]
            for mode_ix in range(4):
                normed_slices[slice_ix][mode_ix] = self._normalize(self.slices_by_slice[slice_ix][mode_ix])
        print('Done')
        return normed_slices

    def _normalize(self, slice):
        b, t = np.percentile(slice, (.5, 99.5))
        slice = np.clip(slice, b, t)
        if np.std(slice) == 0:
            return slice
        else:
            return (slice - np.mean(slice)) / np.std(slice)

    def save_patient(self, reg_norm_n4, patient_num):
        print('Saving scans for patient {}...'.format(patient_num))
        progress.currval = 0
        if reg_norm_n4 == 'norm':
            for slice_ix in progress(range(155)):
                strip = self.normed_slices[slice_ix].reshape(1200, 240)
                if np.max(strip) != 0:
                    strip /= np.max(strip)
                if np.min(strip) <= -1:
                    strip /= abs(np.min(strip))
                io.imsave('NORM_PNG/{0}_{1:04d}.png'.format(patient_num, slice_ix), strip)
        elif reg_norm_n4 == 'reg':
            for slice_ix in progress(range(155)):
                strip = self.slices_by_slice[slice_ix].reshape(1200, 240)
                if np.max(strip) != 0:
                    strip /= np.max(strip)
                io.imsave('Training_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)
        else:
            for slice_ix in progress(range(155)):
                strip = self.normed_slices[slice_ix].reshape(1200, 240)
                if np.max(strip) != 0:
                    strip /= np.max(strip)
                if np.min(strip) <= -1:
                    strip /= abs(np.min(strip))
                io.imsave('n4_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)

    def n4itk_norm(self, path, n_dims=3, n_iters='[20, 20, 10, 5]'):
        output_fn = path[:-4] + '_n.mha'
        process_string = f'python src/n4_bias_correction.py {path} {str(n_dims)} "{n_iters}" {output_fn}'
        subprocess.call(process_string, shell=True)


def save_patient_slices(patients, type):
    for patient_num, path in enumerate(patients):
        a = BrainPipeline(path,n4itk=False, n4itk_apply=False)
        a.save_patient(type, patient_num)


def save_labels(labels):
    progress.currval = 0
    for label_idx in progress(range(len(labels))):
        slices = io.imread(labels[label_idx], plugin='simpleitk')
        print('Slices: ', len(slices))
        for slice_idx in range((len(slices))):
            io.imsave('Labels/{0}_{1:04d}L.png'.format(label_idx, slice_idx), slices[slice_idx])


if __name__ == '__main__':
    labels = glob('Dataset/BRATS2015/training/HGG/**/*more*.mha')
    print('Labels: ', len(labels))
    save_labels(labels)
    patients = glob('Dataset/BRATS2015/training/HGG/**')
    save_patient_slices(patients, 'reg')
    save_patient_slices(patients, 'norm')
    save_patient_slices(patients, 'n4')
