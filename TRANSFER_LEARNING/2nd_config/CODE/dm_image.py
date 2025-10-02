import numpy as np
from numpy.random import RandomState
from os import path
import os
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    Iterator,
)
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import backend as K
import cv2
try:
    import dicom
except ImportError:
    import pydicom as dicom
from dm_preprocess import DMImagePreprocessor as prep
from sklearn.cluster import KMeans
from skimage.feature import blob_dog, blob_log, blob_doh
import time
import sys

data_format = K.image_data_format()
if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3

def to_sparse(y):
    '''Convert labels to sparse format if they are onehot encoded
    '''
    if y.ndim == 1:
        sparse_y = y
    elif y.ndim == 2:
        sparse_y = []
        for r in y:
            label = r.nonzero()[0]
            if len(label) != 1 or r[label[0]] != 1:
                raise ValueError('Expect one-hot encoding for y. '
                                 'Got sample:', r)
            sparse_y.append(label)
        sparse_y = np.concatenate(sparse_y)
    else:
        raise ValueError('Unable to convert y into sparse format.'
                         'Unexpected dimension for y:', y.ndim)
    return sparse_y


def index_balancer(index_array, classes, ratio, rng=None):
    '''Balance an index array according to desired neg:pos ratio
    '''
    if classes is None:
        return index_array
    # this implementation assumes there are only two classes
    pos_idx = np.where(classes==1)[0]
    neg_idx = np.where(classes==0)[0]
    if len(neg_idx) == 0:
        return index_array
    # sample the positive pool such that neg:pos == ratio given all negatives
    # new_nbpos = nb_neg / ratio
    # *note* if new_nbpos >= cur_nbpos, then keep all positives.
    new_pos_nb = int(len(neg_idx)/ratio)
    # *note* if new_nbpos == cur_nbpos, then keep all positives.
    if new_pos_nb < len(pos_idx):
        if rng is None:
            rng = np.random.RandomState()
        pos_idx = pos_idx[rng.choice(len(pos_idx), size=new_pos_nb, replace=False)]
    else:
        print('Index balancer does not take effect:\n'
              'Desired ratio is likely too large given the negative pool size.'
              'Try a smaller ratio instead.')
    return np.concatenate([pos_idx, neg_idx])


def read_resize_img(fname, target_size=None, target_height=None, 
                    target_scale=None, gs_255=False, rescale_factor=None):
    '''Read an image (.png, .jpg, .dcm) and resize it to target size.
    '''
    if target_size is None and target_height is None:
        raise Exception('One of [target_size, target_height] must not be None')
    if path.splitext(fname)[1] == '.dcm':
        img = dicom.read_file(fname).pixel_array
    else:
        if gs_255:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if target_height is not None:
        # make sure resized image is at least target_height
        if img.shape[ROW_AXIS] > target_height:
            # print('Img larger than target height: expected = %s, got = %s' % 
            #         (str(target_height), str(img.shape[ROW_AXIS])))
            target_size = (target_height*img.shape[COL_AXIS]/img.shape[ROW_AXIS], target_height)
        else:
            img_c = np.zeros([target_height, int(target_height*img.shape[COL_AXIS]/img.shape[ROW_AXIS])], 
                             dtype=img.dtype)
            img_c[:img.shape[ROW_AXIS], :img_c.shape[COL_AXIS]] = img
            img = img_c
            target_size = None

    if target_size is not None:
        fx = float(target_size[COL_AXIS-1]) / img.shape[COL_AXIS]
        fy = float(target_size[ROW_AXIS-1]) / img.shape[ROW_AXIS]
        img = cv2.resize(img, (target_size[COL_AXIS-1], target_size[ROW_AXIS-1]), 
                         interpolation=cv2.INTER_CUBIC)
    if target_scale is not None:
        img_max = img.max() if img.max() != 0 else target_scale
        img = img*target_scale/img_max
    if rescale_factor is not None:
        img = img #quitamos el doble rescale factor
    return img.astype('float32')


def read_img_for_pred(fname, target_size, target_scale, label=None, 
                      gs_255=False, rescale_factor=None, dup_3_channels=False, 
                      equalize_hist=False, preprocess=None, 
                      featurewise_center=False, featurewise_mean=0):
    '''Read image files and possibly do histogram equalization.
    '''
    if path.splitext(fname)[1] == '.dcm':
        img = dicom.read_file(fname).pixel_array
    else:
        if gs_255:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    if target_scale is not None:
        img_max = img.max() if img.max() != 0 else target_scale
        img = img*target_scale/img_max
    if equalize_hist:
        img = cv2.equalizeHist(img.astype('uint8'))
    if rescale_factor is not None:
        img = img #quitamos el doble rescale factor
    if preprocess is not None:
        img = preprocess(img)
    if dup_3_channels:
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img] * 3, -1)
    if featurewise_center:
        img -= featurewise_mean
    if label is None:
        return img.astype('float32')
    if isinstance(label, list):
        label = np.array(label)
    if img.ndim == 2 or img.shape[CHANNEL_AXIS] == 1:
        img = np.expand_dims(img, axis=CHANNEL_AXIS)
    return img.astype('float32'), to_categorical(label.astype(int))


def get_roi_patches(img, key_pts, roi_size):
    '''Extract patches centered at each key point with size roi_size.
    '''
    half_roi = roi_size/2
    patch_list = []
    for pt in key_pts:
        x, y = tuple(np.round(pt).astype('int32'))
        patch = img[y-half_roi:y+half_roi, x-half_roi:x+half_roi].copy()
        patch_list.append(patch.astype('float32'))
    return np.stack(patch_list)


def clust_kpts(key_pts, nb_pt=3):
    '''Clustering for grouping key points and return the most probable doamin.
    '''
    if len(key_pts) <= nb_pt:
        return key_pts
    kmeans = KMeans(n_clusters=nb_pt, random_state=13)
    kmeans.fit(key_pts)
    cluster_size = [(c, len(np.where(kmeans.labels_ == c)[0])) for c in range(nb_pt)]
    cluster_size = sorted(cluster_size, key=lambda r: -r[1])
    return key_pts[np.where(kmeans.labels_ == cluster_size[0][0])[0]]


def sweep_img_patches(img, target_scale, patch_size, stride, center=False, equalize_hist=False):
    '''Sweep the image with a sliding window to get patches for the image.
    '''
    if center:
        nb_row = int(np.ceil((img.shape[ROW_AXIS]-patch_size)/float(stride))) + 1
        nb_col = int(np.ceil((img.shape[COL_AXIS]-patch_size)/float(stride))) + 1
        y_gap = int((img.shape[ROW_AXIS] - (nb_row-1)*stride - patch_size)/2.0)
        x_gap = int((img.shape[COL_AXIS] - (nb_col-1)*stride - patch_size)/2.0)
    else:
        nb_row = (img.shape[ROW_AXIS]-patch_size)//stride + 1
        nb_col = (img.shape[COL_AXIS]-patch_size)//stride + 1
        y_gap = 0
        x_gap = 0
    patch_list = []
    for y in range(y_gap, y_gap + nb_row*stride, stride):
        for x in range(x_gap, x_gap + nb_col*stride, stride):
            patch = img[y:y+patch_size, x:x+patch_size].copy()
            if target_scale is not None:
                patch_max = patch.max() if patch.max() != 0 else target_scale
                patch *= target_scale/patch_max
            if equalize_hist:
                patch = cv2.equalizeHist(patch.astype('uint8'))
            patch_list.append(patch.astype('float32'))
    return np.stack(patch_list), nb_row, nb_col


def get_prob_heatmap(img_list, target_height, target_scale, patch_size, stride, 
                     model, batch_size, 
                     featurewise_center=False, featurewise_mean=91.6,
                     preprocess=None, parallelized=False, 
                     equalize_hist=False, verbose=False):
    '''Generate classification probability heatmaps for all images in a list.
    '''
    from dm_multi_gpu import make_parallel
    if parallelized:
        print('Using a multi-GPU model.')
        model = make_parallel(model, 2)[0]
    heatmaps = []
    t0 = time.time()
    for i in range(len(img_list)):
        # pre-process image
        img = img_list[i]
        img = prep.segment_breast(img.astype('uint8'), smooth=True)
        img = prep.batch_segment(img, return_part=True, part_id=1)
        # enlarge the breast region by padding border
        margin_size = 64
        enlarged_img = np.zeros((img.shape[0]+margin_size*2, img.shape[1]+margin_size*2))
        enlarged_img[margin_size:margin_size+img.shape[0], margin_size:margin_size+img.shape[1]] = img
        img = enlarged_img

        # sweep-image patches
        patches, nb_row, nb_col = sweep_img_patches(img, target_scale, patch_size, stride, 
                                                    center=True, equalize_hist=equalize_hist)

        # classification
        X = np.stack([patches]*3, axis=-1) # assume ResNet50 input shape
        if featurewise_center:
            X -= featurewise_mean
        if preprocess is not None:
            X = preprocess(X)
        # Avoid allocating too much memory for prediction
        if X.nbytes > 1e9:
            batch = int(1e9 / X.nbytes * X.shape[0])
        else:
            batch = batch_size
        predict_res = model.predict(X, batch_size=batch, verbose=0)
        if predict_res.ndim == 2 and predict_res.shape[1] > 1:
            predict_res = predict_res[:, 1]
        heatmaps.append(predict_res.reshape((nb_row, nb_col)))

        # verbose
        if verbose and (i+1) % 10 == 0:
            print('   %d images processed..' % (i+1))
    print('Total time spent in generating heatmaps for %d images = %.2f seconds' % 
          (len(img_list), time.time()-t0))
    return heatmaps


def get_index_generator(nb_data, batch_size, shuffle):
    '''Return a generator that maintains indexes for a data list
    '''
    nb_batch = int(np.ceil(nb_data / float(batch_size)))
    # seed = np.random.randint(1e6)
    rng = RandomState()
    def _gen():
        while True:
            index_array = np.arange(nb_data)
            if shuffle:
                rng.shuffle(index_array)
            for batch in range(nb_batch):
                idx = slice(batch*batch_size, min(nb_data, (batch+1)*batch_size))
                yield index_array[idx]
    return _gen()


class DMImgListIterator(Iterator):
    '''Flatten a list of images and convert it into a generator (zero), i.e., - heatmap
    '''
    def __init__(self, 
                 img_list, lab_list, image_data_generator,
                 target_size, target_scale, gs_255=False,
                 class_mode='binary', validation_mode=False,
                 balance_classes=False, all_neg_skip=0.,
                 data_format=None,
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg', verbose=True):
        if data_format is None:
            data_format = K.image_data_format()
        self.validation_mode = validation_mode
        # parameters for validation_only
        self.balance_classes = balance_classes
        self.all_neg_skip = all_neg_skip
        self.class_mode = class_mode
        if class_mode == 'categorical':
            self.nb_class = lab_list.shape[-1]
            self.nb_class_dim = lab_list.ndim - 1
        elif class_mode == 'binary':
            self.nb_class = 2
            self.nb_class_dim = lab_list.ndim
        elif class_mode == 'sparse':
            self.nb_class = lab_list.max() + 1
            self.nb_class_dim = lab_list.ndim
        else:
            raise Exception('Unknown class_mode: ' + class_mode)
        # assign class labels
        self.classes = []
        for l in lab_list:
            if l.ndim == 0:
                self.classes.append(int(l))
            else:
                self.classes.append(l.nonzero()[0][0])
        self.classes = np.array(self.classes)
        # image related attributes
        self.X = img_list
        self.y = lab_list
        self.nb_sample = len(img_list)
        self.image_data_generator = image_data_generator
        self.target_size = target_size
        self.target_scale = target_scale
        self.gs_255 = gs_255
        # utils for data generator
        if seed is None:
            seed = np.random.randint(1e6)
        self.rng = RandomState(seed)
        # meta-data for evaluate studies
        self.exam_cls_list = []
        self.viewname_list = []
        self.exam_list = []
        # init the base class
        super(DMImgListIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)
        if verbose:
            print('DMImgListIterator: %d samples located.' % (self.nb_sample))
            sys.stdout.flush()

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    def next(self):
        return self.__next__()

    def __next__(self):
        # Prepare a balanced batch when called for training
        if self.validation_mode:
            index_array = next(self._get_validation_index_array())
        else:
            index_array = next(self.index_generator)
        # allocate batch memory
        if self.gs_255:
            batch_x = np.zeros((len(index_array), self.target_size[ROW_AXIS-1], 
                                self.target_size[COL_AXIS-1]), dtype=np.float32)
        else:
            batch_x = np.zeros((len(index_array), self.target_size[ROW_AXIS-1], 
                                self.target_size[COL_AXIS-1]), dtype=np.float32)
        batch_y = None
        if self.class_mode is not None:
            if self.class_mode == 'categorical':
                batch_y = np.zeros(len(index_array), dtype='int32')
            elif self.class_mode == 'binary':
                batch_y = np.zeros(len(index_array), dtype='float32')
            elif self.class_mode == 'sparse':
                batch_y = np.zeros(len(index_array), dtype='int32')
        # build batch of image data
        for i, j in enumerate(index_array):
            # exit when all indexes are covered
            # import pdb; pdb.set_trace()
            # i.e., this condition is not necessary though.
            if i >= self.nb_sample:
                break
            fname = self.X[j]
            if self.class_mode is not None:
                # print('Befor conversion %s: %s' % (self.class_mode, str(self.y[j])))
                # safely support cross-entropy and hinge loss with this trick
                if self.class_mode == 'categorical':
                    batch_y[i] = self.y[j].argmax()
                elif self.class_mode == 'binary':
                    batch_y[i] = self.y[j]
                elif self.class_mode == 'sparse':
                    batch_y[i] = self.y[j]
            # read and resize images
            if path.splitext(fname)[1] == '.dcm':
                img = dicom.read_file(fname).pixel_array
            else:
                img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            if self.target_size is not None:
                if self.target_scale is not None:
                    # padding to increase image size so that max of image height equals target_height
                    if img.shape[ROW_AXIS-1] < self.target_size[ROW_AXIS-1]:
                        img_c = np.zeros([self.target_size[ROW_AXIS-1], img.shape[COL_AXIS]], dtype=img.dtype)
                        img_c[:img.shape[ROW_AXIS], :img.shape[COL_AXIS]] = img
                        img = img_c
                img = cv2.resize(img, (self.target_size[COL_AXIS-1], self.target_size[ROW_AXIS-1]),
                                 interpolation=cv2.INTER_CUBIC)
            if self.target_scale is not None:
                img_max = img.max() if img.max() != 0 else self.target_scale
                img = img*self.target_scale/img_max
            # add one dimension if necessary
            if img.ndim == 2:
                # RGB for general deep networks
                pass
            # add image to batch
            batch_x[i] = img.astype(np.float32)
        # standardize images
        if self.gs_255:
            # directly duplicate single channel to rgb
            batch_x = np.stack([batch_x, batch_x, batch_x], axis=CHANNEL_AXIS)
        # apply transformation (standardization + augmentation)
        # Also grayscale image will not be changed (except rescaled multiplied) by transforms.
        for i in range(len(index_array)):
            img = img
        # return blob of X and y
        if self.class_mode is None:
            return batch_x
        else:
            if self.class_mode == 'categorical':
                batch_y = to_categorical(batch_y, num_classes=self.nb_class)
            return batch_x, batch_y

    def _get_validation_index_array(self):
        # balance positive and negative classes according to ratio
        rng = RandomState()
        index_array = np.arange(self.nb_sample)
        # sample subset of positives to maintain a diversity pool
        # index_array = index_array[::2]
        num_neg = len(np.where(self.classes == 0)[0])
        num_pos = len(np.where(self.classes == 1)[0])
        if self.balance_classes:
            if num_pos == 0:
                raise Exception('Cannot do class balancing when there is no positive samples')
            pos_list = index_array[np.where(self.classes == 1)[0]]
            neg_list = index_array[np.where(self.classes == 0)[0]]
            neg_list = neg_list[rng.choice(len(neg_list), size=num_pos, replace=False)]
            index_array = np.concatenate([pos_list, neg_list])
        if self.all_neg_skip > 0:
            # randomly drop out a portion of negatives
            index_array = index_balancer(index_array, self.classes, 1./(1.-self.all_neg_skip))
        self.rng.shuffle(index_array)
        # Yield each batch of indexes
        nb_batch = int(np.ceil(len(index_array)/float(self.batch_size)))
        for batch in range(nb_batch):
            idx = slice(batch*self.batch_size, min(len(index_array), (batch+1)*self.batch_size))
            yield index_array[idx]


class DMExamListIterator(Iterator):
    '''An iterator for a flatten exam list
    '''

    def __init__(self, exam_list, image_data_generator,
                 target_size=(1152, 896), target_scale=4095, gs_255=False, 
                 data_format='default',
                 class_mode='binary', validation_mode=False, prediction_mode=False, 
                 balance_classes=False, all_neg_skip=0.,
                 batch_size=16, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg', verbose=True):

        if data_format == 'default':
            data_format = K.image_data_format()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.target_scale = target_scale
        self.gs_255 = gs_255
        self.data_format = data_format
        # Always gray-scale. Two inputs: CC and MLO.
        if self.data_format == 'channels_last':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.validation_mode = validation_mode
        self.prediction_mode = prediction_mode
        if validation_mode or prediction_mode:
            shuffle = False
            balance_classes = False
            all_neg_skip = 0.
        self.balance_classes = balance_classes
        self.all_neg_skip = all_neg_skip
        self.seed = seed
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.verbose = verbose
        # Convert exam list.
        self.exam_list = exam_list
        self.nb_exam = len(exam_list)
        self.nb_class = 2
        self.err_counter = 0
        # For each exam: 0 => subj id, 1 => exam idx, 2 => exam dat.
        self.classes = [ (e[2]['L']['cancer'], e[2]['R']['cancer']) 
                         for e in exam_list ]
        self.classes = np.array(self.classes)  # (exams, breasts)
        if verbose:
            print('For left breasts, normal=%d, cancer=%d, unimaged/masked=%d.' % 
                (np.sum(self.classes[:, 0] == 0), 
                 np.sum(self.classes[:, 0] == 1), 
                 np.sum(np.isnan(self.classes[:, 0])))
                )
            print('For right breasts, normal=%d, cancer=%d, unimaged/masked=%d.' % 
                (np.sum(self.classes[:, 1] == 0), 
                 np.sum(self.classes[:, 1] == 1), 
                 np.sum(np.isnan(self.classes[:, 1])))
                )

        super(DMExamListIterator, self).__init__(
            self.nb_exam, batch_size, shuffle, seed)


    def next(self):
        with self.lock:
            # initialize batch state
            index_generator = self.index_generator
            # A multi-input model that applies evaluation on both left and right breasts.
            index_array = next(index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        if self.prediction_mode:
            batch_x_cc = []  # a list (of breasts) of lists of image arrays.
            batch_x_mlo = []
            batch_subj = []
            batch_exam = []
        else:
            batch_x_cc = np.zeros( (len(index_array),) + self.image_shape, dtype='float32' )
            batch_x_mlo = np.zeros( (len(index_array),) + self.image_shape, dtype='float32' )


        def draw_img(img_df, exam=None):
            '''Read image(s) based on different modes
            Returns: a single image array or a list of image arrays
            '''
            try:
                if self.prediction_mode:
                    img = []
                    for fname in img_df['filename']:
                        img.append(read_resize_img(
                            fname, self.target_size, 
                            target_scale=self.target_scale, 
                            gs_255=self.gs_255))
                    if len(img) == 0:
                        raise ValueError('empty image dataframe')
                else:
                    if self.validation_mode:
                        fname = img_df['filename'].iloc[0]  # read the canonical view.
                    else:  # training mode.
                        fname = img_df['filename'].sample(1, random_state=rng).iloc[0]
                    img = read_resize_img(
                        fname, self.target_size, target_scale=self.target_scale, 
                        gs_255=self.gs_255)
            except ValueError:
                if self.err_counter < 10:
                    print("Error encountered reading an image dataframe:")
                    print(img_df, "Use a blank image instead.")
                    print("Exam caused trouble:", exam)
                if self.prediction_mode:
                    img = [np.zeros(self.target_size, dtype='float32')]
                else:
                    img = np.zeros(self.target_size, dtype='float32')
                self.err_counter += 1
            return img

        # allocate a temporary hold for censored samples
        tmp_batch_x_cc = []
        tmp_batch_x_mlo = []
        tmp_batch_y = []
        tmp_batch_w = []
        # copy images into general storage for the selected samples
        # current_batch_size should be subj # in this batch * #view (= 2).
        current_batch_size = 0
        # loops
        rng = RandomState(self.seed)
        for i, j in enumerate(index_array):
            # Each row of exam is indexed by: subj idx, exam idx, and exam dat.
            subj, exam, dat = self.exam_list[j]
            # Typical views pairs from each breast.
            # B, L/R; V, CC/MLO (?).
            # The dictionary entries are:
            #   [breast]['filename'] => pds for ['filename', 'width', 'height']
            #                          for one or multiple images (views).
            #   [breast]['cancer']   => 0/1 or NaN.
            #   [breast]['mask']     => pds for file name (full path).
            #   [breast]['bbox']     => bounding box in mm.
            #   [breast]['resolution'] => resolution in mm/pixel.
            # Note: the breast portion may be null (NaN) or empty ([]) — that is, two
            # different flavor of exam data corruption.
            # For a valid stored exam under protection, the typical flow is:
            # - check existence of breast mask file and load if we don't want to
            #   save those images.
            # - read image files; assume breast region already masked out for privacy.
            # - skip this sample (no saved image) if some views are missing;
            #   otherwise check saved image is consistent with mask.
            # - add the sample to the input blob.
            # For a prediction input, read the raw dicom file then do a light-weight
            # pre-processing (# of images might be >2):
            # - segmentation into a bounding box containing breast region.
            # - cropping then re-sizing to a target size.
            # *Note* corruption handling (on non-protected exam only):
            #   - if a mask is missing and saving images is required, then skip.
            #   - if a view is missing for a healthy breast, skip the exam.
            #   - if a view is missing for a cancer breast, print a warning but
            #     allow binary classification if the other breast is okay and the
            #     contra-lateral view of the cancer breast is available.
            #   - if both views are missing for a cancer breast, skip the exam and
            #     print a warning.
            #
            # Returns from skipping current exam:
            # - Use continue here;
            #   i is the counter of current batch; untouched until end of loops.
            #   I have to prepare the mapping idx between loaded exam and batch.
            try:
                exam_dat = dat
                # for prediction, update into raw dicom files. (skipped here)
                # more checks with the mask file should be placed here when
                # those are not protected yet.
                # Note. Some datasets go with fully masked image files thus reading
                # the mask file won't help.
                if self.prediction_mode:
                    subj = subj
                else:
                    subj = subj
                # Missing view handling.
                if np.isnan(exam_dat['L']['cancer']) and np.isnan(exam_dat['R']['cancer']):
                    continue
                # Positive exam (cancer in L or R) is determining.
                pos_left = 1 if exam_dat['L']['cancer'] == 1 else 0
                pos_right = 1 if exam_dat['R']['cancer'] == 1 else 0
                # Make final decision of skipping
                if self.validation_mode:
                    # In validation mode, do not skip positives unless no image is available.
                    pass
                else:
                    # Training: skip some negatives
                    if self.all_neg_skip > 0 and (pos_left+pos_right) == 0:
                        if rng.rand() < self.all_neg_skip:
                            continue

                # Read images (CC and MLO); treat prediction specially
                if self.prediction_mode:
                    # read all images (filenames) for CC and MLO
                    img_cc = draw_img(exam_dat['L']['filename'], exam=(subj, exam, 'L-cc')) \
                             if isinstance(exam_dat['L']['filename'], dict) else draw_img(exam_dat['L'], exam=(subj, exam, 'L-cc'))
                    img_m  = draw_img(exam_dat['L']['filename'], exam=(subj, exam, 'L-mlo')) \
                             if isinstance(exam_dat['L']['filename'], dict) else draw_img(exam_dat['L'], exam=(subj, exam, 'L-mlo'))
                    img_cc_R = draw_img(exam_dat['R']['filename'], exam=(subj, exam, 'R-cc')) \
                               if isinstance(exam_dat['R']['filename'], dict) else draw_img(exam_dat['R'], exam=(subj, exam, 'R-cc'))
                    img_m_R  = draw_img(exam_dat['R']['filename'], exam=(subj, exam, 'R-mlo')) \
                               if isinstance(exam_dat['R']['filename'], dict) else draw_img(exam_dat['R'], exam=(subj, exam, 'R-mlo'))
                else:
                    # training/validation: pick one canonical (or random for train) per view
                    img_cc = draw_img(exam_dat['L'], exam=(subj, exam, 'L-cc'))
                    img_m  = draw_img(exam_dat['L'], exam=(subj, exam, 'L-mlo'))
                    img_cc_R = draw_img(exam_dat['R'], exam=(subj, exam, 'R-cc'))
                    img_m_R  = draw_img(exam_dat['R'], exam=(subj, exam, 'R-mlo'))

                # if prediction, we are building lists of arrays per breast
                if self.prediction_mode:
                    batch_x_cc.append( [np.expand_dims(x, -1) for x in img_cc] + [np.expand_dims(x, -1) for x in img_cc_R] )
                    batch_x_mlo.append( [np.expand_dims(x, -1) for x in img_m] + [np.expand_dims(x, -1) for x in img_m_R] )
                    batch_subj.append(subj)
                    batch_exam.append(exam)
                    current_batch_size += 1
                else:
                    # place into batch tensors (one per view)
                    L_cc = np.expand_dims(img_cc, -1)
                    L_m  = np.expand_dims(img_m, -1)
                    R_cc = np.expand_dims(img_cc_R, -1)
                    R_m  = np.expand_dims(img_m_R, -1)

                    # For simplicity, use left-CC and left-MLO; extend to bilateral if needed
                    if self.data_format == 'channels_first':
                        L_cc = L_cc.transpose(2,0,1)
                        L_m  = L_m.transpose(2,0,1)
                    batch_x_cc[current_batch_size] = L_cc
                    batch_x_mlo[current_batch_size] = L_m
                    # label is positive if either side is positive
                    y = 1 if (pos_left + pos_right) > 0 else 0
                    tmp_batch_y.append(y)
                    current_batch_size += 1

            except Exception as e:
                # data corruption handling
                if self.err_counter < 10:
                    print("Exception caught when processing exam: ", (subj, exam))
                    print("Error:", e)
                self.err_counter += 1
                continue

        # Prepare output blobs
        if self.prediction_mode:
            # Nested lists; let the model code handle aggregation.
            return [batch_x_cc, batch_x_mlo], None
        else:
            batch_x_cc = batch_x_cc[:current_batch_size]
            batch_x_mlo = batch_x_mlo[:current_batch_size]
            batch_y = np.array(tmp_batch_y[:current_batch_size], dtype='int32')
            if self.class_mode == 'categorical':
                batch_y = to_categorical(batch_y, num_classes=self.nb_class)
            return [batch_x_cc, batch_x_mlo], batch_y
class DMNumpyArrayIterator(Iterator):

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, auto_batch_balance=True, no_pos_skip=0.,
                 balance_classes=0., preprocess=None, shuffle=False, seed=None,
                 data_format='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if data_format == 'default':
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())
        if self.x.ndim != 4:
            raise ValueError('Input data in `DMNumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('DMNumpyArrayIterator is expecting images '
                             'with 1, 3 or 4 channels.')
        if y is not None:
            self.y = np.asarray(y)
            if self.y.ndim != 2:
                raise ValueError('`y` should be a 2D array, got:', self.y.shape)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.batch_size = batch_size
        self.auto_batch_balance = auto_batch_balance
        self.no_pos_skip = no_pos_skip
        self.balance_classes = balance_classes
        self.preprocess = preprocess
        self.shuffle = shuffle
        self.seed = seed

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        super(DMNumpyArrayIterator, self).__init__(self.x.shape[0], batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)

        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            if self.preprocess is not None:
                x = self.preprocess(x)
            batch_x[i] = x
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class DMDirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), target_scale=None, gs_255=False,
                 equalize_hist=False, rescale_factor=None,
                 dup_3_channels=False, data_format='default',
                 classes=None, class_mode='categorical', 
                 auto_batch_balance=False, batch_size=32, 
                 preprocess=None, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False):
        '''
        Args:
            dup_3_channels: boolean, whether duplicate the input image onto 3 
                channels or not. This can be useful when using pretrained models 
                from databases such as ImageNet.
        '''
        if data_format == 'default':
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.target_scale = target_scale
        self.gs_255 = gs_255
        self.equalize_hist = equalize_hist
        self.rescale_factor = rescale_factor
        self.dup_3_channels = dup_3_channels
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.auto_batch_balance = auto_batch_balance
        self.preprocess = preprocess
        self.data_format = data_format
        channels = 3 if self.dup_3_channels else 1
        self.image_shape = self.target_size + (channels,) if data_format == 'channels_last' \
            else (channels,) + self.target_size

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'pgm', 'tif', 'tiff'}
        # first, count the number of samples and classes
        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.samples = 0
        self.filenames = []
        self.classes = []

        # Build an index of the images in the different class subfolders
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in os.walk(subpath):
                for fname in sorted(files):
                    is_valid = (fname.split('.')[-1].lower() in white_list_formats)
                    if is_valid:
                        self.samples += 1
                        self.filenames.append(os.path.join(root, fname))
                        self.classes.append(self.class_indices[subdir])
                if not follow_links:
                    break

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))
        self.classes = np.array(self.classes)
        super(DMDirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        # allocate
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = read_img_for_pred(
                fname, (self.target_size[1], self.target_size[0]), self.target_scale,
                gs_255=self.gs_255, rescale_factor=self.rescale_factor,
                dup_3_channels=False, equalize_hist=self.equalize_hist,
            )

            # Duplicate channel if needed
            if self.dup_3_channels:
                img = np.stack([img] * 3, -1)
            else:
                img = np.expand_dims(img, -1)
            # preprocess
            if self.preprocess is not None:
                img = self.preprocess(img)
            # place
            if self.data_format == 'channels_first':
                img = img.transpose(2,0,1)
            batch_x[i] = img
        # labels if available
        if self.class_mode is None:
            return batch_x
        batch_y = to_categorical(self.classes[index_array], num_classes=self.num_class) \
                  if self.class_mode == 'categorical' else self.classes[index_array]
        return batch_x, batch_y


class DMImageDataGenerator(ImageDataGenerator):
    '''Image data generator for digital mammography
    '''

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=10,          # rotación suave ±10°
                 width_shift_range=0.05,     # desplazamiento horizontal 5%
                 height_shift_range=0.05,    # desplazamiento vertical 5%
                 shear_range=0.,             # evitar deformaciones
                 zoom_range=0.1,             # zoom ±10%
                 channel_shift_range=0.,     # sin cambios de contraste iniciales
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=True, 
                 vertical_flip=False,
                 rescale=None,
                 data_format='default'):

        if data_format == 'default':
            data_format = K.image_data_format()
        super(DMImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            data_format=data_format
        )
        self.data_format = data_format

    # ---- new helpers matching original API ----
    def flow_from_exam_list(self, exam_list, **kwargs):
        return DMExamListIterator(exam_list, self, **kwargs)

    def flow_from_directory(self, directory, **kwargs):
        return DMDirectoryIterator(directory, self, **kwargs)

    def flow_from_img_list(self, img_list, lab_list, **kwargs):
        return DMImgListIterator(img_list, lab_list, self, **kwargs)

    def flow(self, x, y=None, **kwargs):
        return DMNumpyArrayIterator(x, y, self, **kwargs)
