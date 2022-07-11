import json
from sklearn.metrics import classification_report
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.core import Activation, Dropout, Flatten, Dense, Reshape
from keras.regularizers import l1_l2
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from patch_library import PatchLibrary


class SegmentationModel(object):
    def __init__(self, n_epoch=10, n_chan=4, batch_size=128, loaded_model=False, architecture='single', w_reg=0.01,
                 n_filters=None, k_dims=None, activation='relu'):
        if k_dims is None:
            k_dims = [7, 5, 5, 3]
        if n_filters is None:
            n_filters = [64, 128, 128, 128]
        self.n_epoch = n_epoch
        self.n_chan = n_chan
        self.batch_size = batch_size
        self.loaded_model = loaded_model
        self.architecture = architecture
        self.w_reg = w_reg
        self.n_filters = n_filters
        self.k_dims = k_dims
        self.activation = activation
        if not self.loaded_model:
            if self.architecture == 'two_path':
                self.model_comp = self.comp_two_path()
            elif self.architecture == 'dual':
                self.model_comp = self.comp_double()
            else:
                self.model_comp = self.compile_model()
        else:
            model = str(input('Which model should I load? '))
            self.model_comp = self.load_model_weights(model)

    def compile_model(self):
        print('Compiling single mode: ')
        single = Sequential()

        single.add(Convolution2D(self.n_filters[0], self.k_dims[0], self.k_dims[0], border_mode='valid',
                                 W_regularizer=l1_l2(l1=self.w_reg, l2=self.w_reg),
                                 input_shape=(self.n_chan, 33, 33)))
        single.add(Activation(self.activation))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        single.add(Dropout(0.5))

        single.add(Convolution2D(self.n_filters[1], self.k_dims[1], self.k_dims[1], border_mode='valid',
                                 W_regularizer=l1_l2(l1=self.w_reg, l2=self.w_reg), input_shape=(self.n_chan, 33, 33)))
        single.add(Activation(self.activation))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        single.add(Dropout(0.5))

        single.add(Convolution2D(self.n_filters[2], self.k_dims[2], self.k_dims[2], border_mode='valid',
                                 W_regularizer=l1_l2(l1=self.w_reg, l2=self.w_reg), input_shape=(self.n_chan, 33, 33)))
        single.add(Activation(self.activation))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        single.add(Dropout(0.5))

        single.add(Convolution2D(self.n_filters[3], self.k_dims[3], self.k_dims[3], border_mode='valid',
                                 W_regularizer=l1_l2(l1=self.w_reg, l2=self.w_reg), input_shape=(self.n_chan, 33, 33)))
        single.add(Activation(self.activation))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        single.add(Dropout(0.25))

        single.add(Flatten())
        single.add(Dense(5))
        single.add(Activation('softmax'))

        sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)
        single.compile(loss='categorical_crossentropy', optimizer='sgd')
        print('Done compiling single architecture')
        return single

    def comp_two_path(self):
        """
        Graph module has been removed from Keras, so we will need to use the functional API insteas
        :return:
        """
        pass

    def comp_double(self):
        print('Compiling double module')
        single = Sequential()
        single.add(Convolution2D(64, 7, 7, border_model='valid', W_regularizer=l1_l2(l1=0.01, l2=0.01),
                                 input_shape=(4, 33, 33)))

        single.add(Activation('relu'))
        single(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        single.add(Dropout(0.5))

        single.add(Convolution2D(nb_filter=128, nb_row=5, nb_col=5, activation='relu', border_mode='valid',
                                 W_regularizer=l1_l2(l1=0.01, l2=0.01)))

        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        single.add(Dropout(0.5))

        single.add(Convolution2D(nb_filter=256, nb_row=5, nb_col=5, activation='relu',
                                 border_mode='valid', W_regularizer=l1_l2(l1=0.01, l2=0.01)))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        single.add(Dropout(0.5))

        single.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_valid='valid',
                                 W_regularizer=l1_l2(l1=0.01, l2=0.01)))
        single.add(Dropout(0.25))
        single.add(Flatten())

        five = Sequential()
        five.add(Reshape(100, 1), input_shape=(4, 5, 4))
        five.add(Flatten())
        """
        We need to add the MaxoutDense via Functional API
        """
        #five.add((128, nb_features=5))

    def load_model_weights(self, model_name):
        print('Loading model: {}'.format(model_name))
        model = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        with open(model) as f:
            m = f.__next__()
        model_comp = model_from_json(json.loads(m))
        model_comp.load_weights(weights)
        print('Done loading weight')
        return model_comp

    def fit_model(self, X_train, y_train, X5_train=None, save=True):
        Y_train = np_utils.to_categorical(y_train, 5)
        shuffle = zip(X_train, Y_train)
        np.random.shuffle(shuffle)
        X_train = np.array([shuffle[i][0] for i in range(len(shuffle))])
        Y_train = np.array([shuffle[i][1] for i in range(len(shuffle))])
        checkpointer = ModelCheckpoint(filepath="./check/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
        if self.architecture == 'dual':
            self.model_comp.fit([X5_train, Y_train], Y_train, batch_size=self.batch_size,
                                nb_epoch=self.n_epoch, validation_split=0.1, show_accuracy=True,
                                verbose=1, callbacks=[checkpointer])
        elif self.architecture == 'two_path':
            data = {'input': X_train, 'output': Y_train}
            self.model_comp.fit(data, batch_size=self.batch_size, nb_epoch=self.n_epoch, validation_split=0.1,
                                show_accuracy=True, verbose=1, callbacks=[checkpointer])
        else:
            self.model_comp.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.n_epoch,
                                validation_split=0.1, show_accuracy=True, verbose=1, callbacks=[checkpointer])

    def save_model(self, model_name):
        model = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        json_string = self.model_comp.to_json()
        self.model_comp.save_weights(weights)
        with open(model, 'w') as f:
            json.dump(json_string)

    def class_report(self, X_test, y_test):
        y_pred = self.model_load.predict_class(X_test)
        print(classification_report(y_pred, y_test))

    def predict_image(self, test_image, show=False):
        imgs = io.imread(test_image).astype('float').reshape(5, 240, 240)
        plist = []
        for img in imgs[:-1]:
            if np.max(img) != 0:
                img /= np.max(img)
            p = extract_patches_2d(img, (33, 33))
            plist.append(p)
        patches = np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))
        full_pred = self.model_comp.predict_classes(patches)
        fp1 = full_pred.reshape(208, 208)
        if show:
            io.imshow(fp1)
            plt.show()
        else:
            return fp1

    def show_segmented_image(self, test_img, modality='t1c', show=False):
        modes = {'flair': 0, 't1': 1, 't1c': 2, 't2': 3}
        segmentation = self.predict_image(test_img, show=True)
        img_mask = np.pad(segmentation, (16, 16), mode='edge')
        ones = np.argwhere(img_mask == 1)
        twos = np.argwhere(img_mask == 2)
        threes = np.argwhere(img_mask == 3)
        fours = np.argwhere(img_mask == 4)
        test_im = io.imread(test_img)
        test_back = test_im.reshape(5, 240, 240)[-2]
        gray_img = img_as_float(test_back)
        image = adjust_gamma(color.gray2rgb(gray_img), 0.65)
        sliced_image = image.copy()
        red_multiplier = [1, 0.2, 0.2]
        yellow_multiplier = [1, 1, 0.25]
        green_multiplier = [0.35, 0.75, 0.25]
        blue_multiplier = [0, 0.25, 0.9]
        for i in range(len(ones)):
            sliced_image[ones[i][0]][ones[i][1]] = red_multiplier
        for i in range(len(twos)):
            sliced_image[twos[i][0]][twos[i][1]] = green_multiplier
        for i in range(len(threes)):
            sliced_image[threes[i][0]][threes[i][1]] = blue_multiplier
        for i in range(len(fours)):
            sliced_image[fours[i][0]][fours[i][1]] = yellow_multiplier

        if show:
            io.imshow(sliced_image)
            plt.show()
        else:
            return sliced_image

    def get_dice_coef(self, test_img, label):
        segmentation = self.predict_image(test_img)
        seg_full = np.pad(segmentation, (16, 16), mode='edge')
        gt = io.imread(label).astype(int)
        total = (len(np.argwhere(seg_full == gt)) * 2.0) / (2 * 240 * 240)


def unique_rows(a):
    pass

if __name__ == '__main__':
    training_data = glob('../Dataset/BRATS2015/training/HGG/**')
    print('Trainint data: ', training_data)
    patches = PatchLibrary((33, 33), training_data, 100)
    print('Patches before segmentation: ', patches)
    X, y = patches.make_training_patches()
    model = SegmentationModel()
    model.fit_model(X, y)
    model.save_model('models/example')