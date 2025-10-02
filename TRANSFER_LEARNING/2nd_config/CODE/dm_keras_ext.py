import sys
import os
import math
import time
import random
import numpy as np
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from sklearn.metrics import roc_curve, confusion_matrix

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Input, Activation, Dropout, Dense, Flatten,
    GlobalAveragePooling2D, MaxPooling2D
)
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import (
    SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
)
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")  # backend sin display para guardar figuras

# ------------------------------------------------------------------
# Compat: flip_axis sin depender de keras.preprocessing.image
# ------------------------------------------------------------------
def flip_axis(x, axis):
    return np.flip(x, axis=axis)

# Ejes según data_format
if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3


# ===========================================================
# Flips H/V (como en el original)
# ===========================================================
def flip_all_img(X):
    """Perform horizontal and vertical flips for a 4-D image tensor."""
    if K.image_data_format() == 'channels_last':
        row_axis = 1
        col_axis = 2
    else:
        row_axis = 2
        col_axis = 3
    X_h = flip_axis(X, col_axis)
    X_v = flip_axis(X, row_axis)
    X_h_v = flip_axis(X_h, row_axis)
    return [X, X_h, X_v, X_h_v]

# ===========================================================
# Helpers de métricas y threshold óptimo
# ===========================================================
def _bin_from_onehot(y):
    # y puede venir (N,2) one-hot o (N,)
    return y[:, 1].astype(int) if (y.ndim == 2 and y.shape[1] == 2) else y.astype(int)

def _score_from_logits(y_pred):
    # y_pred puede ser (N,2) softmax o (N,) score binario
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        return y_pred[:, 1]
    return y_pred.ravel()

def find_best_threshold(y_true, y_score):
    """
    Threshold que maximiza G-mean = sqrt(sensibilidad * especificidad).
    Retorna: thr, sens, spec
    """
    yb = _bin_from_onehot(y_true)
    s  = _score_from_logits(y_score)
    from sklearn.metrics import roc_curve
    fpr, tpr, ths = roc_curve(yb, s)  # thresholds alineados con tpr/fpr
    spec = 1.0 - fpr
    g = np.sqrt(tpr * spec)
    if len(g) == 0:
        return 0.5, 0.0, 0.0
    idx = int(np.nanargmax(g))
    return float(ths[idx]), float(tpr[idx]), float(spec[idx])


def bootstrap_auc_ci(y_true, y_score, n_boot=2000, alpha=0.95, seed=13):
    """
    Bootstrap para IC del AUC (percentiles).
    Retorna: (auc, lo, hi)
    """
    yb = _bin_from_onehot(y_true)
    s  = _score_from_logits(y_score)
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(yb)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            aucs.append(roc_auc_score(yb[idx], s[idx]))
        except Exception:
            pass
    aucs = np.array(aucs) if len(aucs) else np.array([0.0])
    lo = np.percentile(aucs, (1.0 - alpha) / 2.0 * 100.0)
    hi = np.percentile(aucs, (1.0 + alpha) / 2.0 * 100.0)
    auc = roc_auc_score(yb, s) if n > 0 else 0.0
    return float(auc), float(lo), float(hi)

def compute_metrics_block(y_true, y_pred_scores, thr=None):
    """
    Calcula AUC (+IC95), acc, sens, spec; si thr=None lo optimiza por Youden.
    Retorna dict con métricas y threshold usado.
    """
    yb = _bin_from_onehot(y_true)
    s  = _score_from_logits(y_pred_scores)
    if thr is None:
        thr, sens_thr, spec_thr = find_best_threshold(y_true, y_pred_scores)
    else:
        sens_thr = spec_thr = None
    yhat = (s >= thr).astype(int)

    # confusion
    tn, fp, fn, tp = confusion_matrix(yb, yhat, labels=[0,1]).ravel()
    acc  = (tp + tn) / max(tn + fp + fn + tp, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    auc, lo, hi = bootstrap_auc_ci(y_true, y_pred_scores)

    return dict(
        threshold=thr,
        accuracy=float(acc),
        sensitivity=float(sens),
        specificity=float(spec),
        auc=float(auc),
        auc_lo=float(lo),
        auc_hi=float(hi),
        sens_at_thr=float(sens_thr) if sens_thr is not None else float(sens),
        spec_at_thr=float(spec_thr) if spec_thr is not None else float(spec),
    )

def _ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    return d
def save_plots_from_history(out_dir, stage_name, history_dict):
    """
    Guarda PNGs de loss, accuracy, AUC, sensibilidad y especificidad
    usando las claves disponibles en history (robusto a 'acc'/'accuracy', etc.).
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    def _g(keys):
        ks = [keys] if isinstance(keys, str) else list(keys)
        for k in ks:
            v = history_dict.get(k)
            if v is not None and len(v) > 0:
                return list(v)
        return None

    def _plot(key_opts, title, fname):
        tr = _g(key_opts)
        vk = ['val_' + key_opts] if isinstance(key_opts, str) else ['val_' + k for k in key_opts]
        vl = _g(vk)
        if tr is None and vl is None:
            return
        ep = range(1, (len(tr) if tr is not None else len(vl)) + 1)
        plt.figure()
        if tr is not None: plt.plot(ep, tr, label='train')
        if vl is not None: plt.plot(ep, vl, label='val')
        plt.title(f'{title} - {stage_name}')
        plt.xlabel('Epoch'); plt.legend()
        fpath = os.path.join(out_dir, f'{stage_name}_{fname}.png')
        plt.savefig(fpath, dpi=160); plt.close()
        print(f'  ✔ {fpath}')

    _plot('loss',                 'Loss',         'loss')
    _plot(['accuracy','acc'],     'Accuracy',     'accuracy')
    _plot(['auc','AUC'],          'AUC',          'auc')
    _plot(['sensitivity','Sens'], 'Sensitivity',  'sensitivity')
    _plot(['specificity','Spec'], 'Specificity',  'specificity')

# ===========================================================
# Carga robusta de modelos
# ===========================================================
def robust_load_model(filepath, custom_objects=None):
    try:
        model = load_model(filepath, custom_objects=custom_objects)
    except ValueError:
        import h5py
        f = h5py.File(filepath, 'r+')
        if 'optimizer_weights' in f:
            del f['optimizer_weights']
        f.close()
        model = load_model(filepath, custom_objects=custom_objects)
    return model


# ===========================================================
# Cargar dataset completo a RAM desde un generator (como original)
# ===========================================================
def load_dat_ram(generator, nb_samples):
    samples_seen = 0
    X_list, y_list, w_list = [], [], []
    while samples_seen < nb_samples:
        # tf.keras DirectoryIterator es indexable; pero .__next__ también funciona
        try:
            blob_ = next(generator)
        except TypeError:
            # fallback si es Sequence puro
            idx = samples_seen // generator.batch_size
            blob_ = generator[idx]
        try:
            X, y, w = blob_
            w_list.append(w)
        except ValueError:
            X, y = blob_
        X_list.append(X)
        y_list.append(y)
        samples_seen += len(y)

    try:
        data_set = (np.concatenate(X_list), np.concatenate(y_list), np.concatenate(w_list))
    except ValueError:
        data_set = (np.concatenate(X_list), np.concatenate(y_list))

    if len(data_set[0]) != nb_samples:
        raise Exception('Load data into RAM error')

    return data_set


# ===========================================================
# Arquitectura Yaroslav (patch-classifier)
# ===========================================================
def Yaroslav(input_shape=None, classes=5):
    """Instantiates the Yaroslav's winning architecture for patch classifiers."""
    if input_shape is None:
        if K.image_data_format() == 'channels_last':
            input_shape = (None, None, 1)
        else:
            input_shape = (1, None, None)
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(256, (3, 3), padding='same', name='block5_conv1')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Block 6
    x = Conv2D(512, (3, 3), padding='same', name='block6_conv1')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block6_conv2')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)

    # Classification block
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='yaroslav')
    return model


# ===========================================================
# get_dl_model (API del original: retorna model, preprocess_input, top_layer_nb)
# ===========================================================
def get_dl_model(net, nb_class=3, use_pretrained=True, resume_from=None,
                 top_layer_nb=None, weight_decay=.01, hidden_dropout=.0, **kw_args):
    """Load existing DL model or create it from new. Retorna (model, preprocess_input, top_layer_nb)."""
    if net == 'resnet50':
        from tensorflow.keras.applications.resnet50 import ResNet50 as NNet, preprocess_input
        top_layer_nb = 162 if top_layer_nb is None else top_layer_nb
    elif net == 'vgg16':
        from tensorflow.keras.applications.vgg16 import VGG16 as NNet, preprocess_input
        top_layer_nb = 15 if top_layer_nb is None else top_layer_nb
    elif net == 'vgg19':
        from tensorflow.keras.applications.vgg19 import VGG19 as NNet, preprocess_input
        top_layer_nb = 17 if top_layer_nb is None else top_layer_nb
    elif net == 'xception':
        from tensorflow.keras.applications.xception import Xception as NNet, preprocess_input
        top_layer_nb = 126 if top_layer_nb is None else top_layer_nb
    elif net in ('inception', 'inception_v3'):
        from tensorflow.keras.applications.inception_v3 import InceptionV3 as NNet, preprocess_input
        top_layer_nb = 194 if top_layer_nb is None else top_layer_nb
    elif net == 'yaroslav':
        top_layer_nb = None
        preprocess_input = None
    else:
        raise Exception("Requested model is not available: " + str(net))

    weights = 'imagenet' if use_pretrained else None

    if resume_from is not None:
        print("Loading existing model state.", end="")
        sys.stdout.flush()
        model = load_model(resume_from)
        print(" Done.")
    elif net == 'yaroslav':
        model = Yaroslav(classes=nb_class)
    else:
        print("Loading %s," % (net), end="")
        sys.stdout.flush()
        base_model = NNet(weights=weights, include_top=False, input_shape=None, pooling='avg')
        x = base_model.output
        if hidden_dropout > 0.:
            x = Dropout(hidden_dropout)(x)
        preds = Dense(nb_class, activation='softmax', kernel_regularizer=l2(weight_decay))(x)
        model = Model(inputs=base_model.input, outputs=preds)
        print(" Done.")

    return model, preprocess_input, top_layer_nb


# ===========================================================
# Optimizer factory (misma API; usa learning_rate kw en TF2)
# ===========================================================
def create_optimizer(optim_name, lr):
    kw = dict(learning_rate=lr)
    if optim_name == 'sgd':
        return SGD(momentum=.9, nesterov=True, **kw)
    elif optim_name == 'rmsprop':
        return RMSprop(**kw)
    elif optim_name == 'adagrad':
        return Adagrad(**kw)
    elif optim_name == 'adadelta':
        return Adadelta(**kw)
    elif optim_name == 'adamax':
        return Adamax(**kw)
    elif optim_name == 'adam':
        return Adam(**kw)
    elif optim_name == 'nadam':
        return Nadam(**kw)
    else:
        raise Exception('Unknown optimizer name: ' + str(optim_name))

def do_3stage_training(
    model, org_model, train_generator, validation_set,
    validation_steps, best_model_out, steps_per_epoch,
    top_layer_nb=None, net=None,
    nb_epoch=0, top_layer_epochs=0, all_layer_epochs=0,
    use_pretrained=True, optim='adam', init_lr=.001,
    top_layer_multiplier=.01, all_layer_multiplier=.0001,
    es_patience=5, lr_patience=2, auto_batch_balance=True,
    nb_class=2, pos_cls_weight=1., neg_cls_weight=1.,
    nb_worker=1,
    # 🔽 NUEVO: dropout para stage2 y (ya existente) para stage3
    hidden_dropout=0,            # stage 2
    weight_decay2=0.015, hidden_dropout2=0.4,  # stage 3
    auc_checkpointer=None, test_generator=None,
    extra_callbacks=None
):
    """
    3-stage training con plots por stage (solo desde logs de Keras):
      - loss (train/val)
      - accuracy (train/val)
      - AUC (train/val)
      - sensitivity (train/val)  *si compilas con DMMetrics.sensitivity
      - specificity (train/val)  *si compilas con DMMetrics.specificity
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
    from tensorflow.keras.metrics import AUC as KerasAUC
    from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Conv2D, SeparableConv2D, DepthwiseConv2D
    from tensorflow.keras.regularizers import l2

    out_dir = os.path.dirname(best_model_out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ---------- helpers ----------
    def _metrics_for_compile():
        mets = ['accuracy', KerasAUC(name='auc')]
        if nb_class == 2:
            mets += [DMMetrics.sensitivity, DMMetrics.specificity]
        return mets

    

    class StageLogger(Callback):
        def __init__(self, e1, e2, e3):
            super().__init__()
            self.e1, self.e2, self.e3 = int(e1), int(e2), int(e3)
            self.global_epoch = -1
        def on_epoch_end(self, epoch, logs=None):
            self.global_epoch += 1
            logs = logs or {}
            if self.global_epoch < self.e1: stage = 1
            elif self.global_epoch < (self.e1 + self.e2): stage = 2
            else: stage = 3
            msg = [
                f"Stage {stage} | epoch {self.global_epoch+1}",
                f"loss={logs.get('loss', float('nan')):.4f}",
                f"acc={logs.get('accuracy', float('nan')):.4f}",
                f"val_loss={logs.get('val_loss', float('nan')):.4f}",
                f"val_acc={logs.get('val_accuracy', float('nan')):.4f}",
                f"val_auc={logs.get('val_auc', float('nan')):.4f}",
            ]
            if 'val_sensitivity' in logs: msg.append(f"val_sens={logs.get('val_sensitivity', float('nan')):.4f}")
            if 'val_specificity' in logs: msg.append(f"val_spec={logs.get('val_specificity', float('nan')):.4f}")
            print(", ".join(msg))

    def _apply_reg_and_dropout(mdl, wd=None, dr=None, trainable_only=True):
        """Ajusta L2 y tasa de Dropout en capas entrenables del modelo org_model."""
        n_do, n_reg = 0, 0
        for l in mdl.layers:
            if trainable_only and not l.trainable:
                continue
            # Dropout
            if dr is not None and isinstance(l, Dropout):
                try:
                    l.rate = float(dr)
                    n_do += 1
                except Exception:
                    pass
            # L2 en conv/dense
            if wd is not None and isinstance(l, (Dense, Conv2D, SeparableConv2D, DepthwiseConv2D)):
                try:
                    l.kernel_regularizer = l2(wd)
                    n_reg += 1
                except Exception:
                    pass
        if dr is not None: print(f"[regularization] Dropout rate -> {dr} en {n_do} capas Dropout (entrenables={trainable_only})")
        if wd is not None: print(f"[regularization] L2 -> {wd} en {n_reg} capas conv/dense (entrenables={trainable_only})")

    # callbacks base
    extra_callbacks = list(extra_callbacks) if extra_callbacks is not None else []
    early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=lr_patience, min_lr=1e-6, verbose=1)
    stage_logger = StageLogger(nb_epoch, top_layer_epochs, all_layer_epochs)

    if auc_checkpointer is None:
        checkpointer = DMAucModelCheckpoint(
            best_model_out,
            val_data=validation_set,
            test_samples=(None if isinstance(validation_set, tuple) else validation_steps),
            batch_size=(1 if isinstance(validation_set, tuple) else None),
            test_augment=False,
            train_data=train_generator,
            test_data=test_generator,
            out_dir=out_dir,
            class_names=None
        )
    else:
        checkpointer = auc_checkpointer

    callbacks_core = [early_stopping, checkpointer, reduce_lr, stage_logger]

    # class_weight
    if auto_batch_balance:
        class_weight = None
    elif nb_class == 2:
        class_weight = {0: 1.0, 1: pos_cls_weight}
    elif nb_class == 3:
        class_weight = {0: 1.0, 1: pos_cls_weight, 2: neg_cls_weight}
    else:
        class_weight = None

    loss_history, acc_history = [], []

    # ===== Stage 1 =====
    if nb_epoch > 0:
        print("[Stage 1] Entrenando solo la densa final…")
        if use_pretrained:
            for layer in org_model.layers[:-1]:
                layer.trainable = False
        model.compile(optimizer=create_optimizer(optim, init_lr),
                      loss='categorical_crossentropy', metrics=_metrics_for_compile())
        hist1 = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            class_weight=class_weight,
            validation_data=validation_set,
            validation_steps=(validation_steps if not isinstance(validation_set, tuple) else None),
            validation_batch_size=1,
            callbacks=callbacks_core + extra_callbacks,
            workers=nb_worker, verbose=1
        )
        save_plots_from_history(out_dir, "stage1", hist1.history)
        loss_history += list(hist1.history.get('val_loss', []) or [])
        acc_history  += list(hist1.history.get('val_accuracy', []) or [])
    else:
        print("[Stage 1] Omitido (nb_epoch=0).")

    # ===== Stage 2 =====
    if use_pretrained and top_layer_epochs > 0:
        print(f"[Stage 2] Fine-tuning top layers (top_layer_nb={top_layer_nb})…")
        for l in org_model.layers[:top_layer_nb]: l.trainable = False
        for l in org_model.layers[top_layer_nb:]: l.trainable = True
        # BN en inference
        for l in org_model.layers:
            if isinstance(l, BatchNormalization): l.trainable = False
        # 🔽 aplica dropout y L2 a capas ENTRENABLES en stage2
        _apply_reg_and_dropout(org_model, wd=weight_decay2, dr=hidden_dropout, trainable_only=True)

        model.compile(optimizer=create_optimizer(optim, init_lr * top_layer_multiplier),
                      loss='categorical_crossentropy', metrics=_metrics_for_compile())
        hist2 = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=top_layer_epochs,
            class_weight=class_weight,
            validation_data=validation_set,
            validation_steps=(validation_steps if not isinstance(validation_set, tuple) else None),
            validation_batch_size=1,
            callbacks=callbacks_core + extra_callbacks,
            workers=nb_worker, verbose=1
        )
        save_plots_from_history(out_dir, "stage2", hist2.history)
        loss_history += list(hist2.history.get('val_loss', []) or [])
        acc_history  += list(hist2.history.get('val_accuracy', []) or [])
    else:
        print("[Stage 2] Omitido.")

    # ===== Stage 3 =====
    if use_pretrained and all_layer_epochs > 0:
        print("[Stage 3] Fine-tuning all layers…")
        for l in org_model.layers[:top_layer_nb]: l.trainable = True
        for l in org_model.layers:
            if isinstance(l, BatchNormalization): l.trainable = False
        # 🔽 aplica dropout y L2 a capas ENTRENABLES en stage3
        _apply_reg_and_dropout(org_model, wd=weight_decay2, dr=hidden_dropout2, trainable_only=True)

        model.compile(optimizer=create_optimizer(optim, init_lr * all_layer_multiplier),
                      loss='categorical_crossentropy', metrics=_metrics_for_compile())
        hist3 = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=all_layer_epochs,
            class_weight=class_weight,
            validation_data=validation_set,
            validation_steps=(validation_steps if not isinstance(validation_set, tuple) else None),
            validation_batch_size=1,
            callbacks=callbacks_core + extra_callbacks,
            workers=nb_worker, verbose=1
        )
        save_plots_from_history(out_dir, "stage3", hist3.history)

        loss_history += list(hist3.history.get('val_loss', []) or [])
        acc_history  += list(hist3.history.get('val_accuracy', []) or [])
    else:
        print("[Stage 3] Omitido.")

    return model, np.array(loss_history), np.array(acc_history)

# ===========================================================
# 2-stage training (whole images) — usado por autores
# ===========================================================
def do_2stage_training(model, org_model, train_generator, validation_set,
                       validation_steps, best_model_out, steps_per_epoch,
                       top_layer_nb=None, nb_epoch=10, all_layer_epochs=0,
                       optim='sgd', init_lr=.01, all_layer_multiplier=.1,
                       es_patience=5, lr_patience=2, auto_batch_balance=True,
                       nb_class=2, pos_cls_weight=1., neg_cls_weight=1., nb_worker=1,
                       auc_checkpointer=None,
                       weight_decay=.0001, hidden_dropout=.0,
                       weight_decay2=.0001, hidden_dropout2=.0,test_generator=None):
    """2-stage DL model training (for whole images) — fidelidad con original."""

    if top_layer_nb is None and nb_epoch > 0:
        raise Exception('top_layer_nb must be specified when nb_epoch > 0')

    # Callbacks / class_weight
    early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1, restore_best_weights=True)
    checkpointer = ModelCheckpoint(best_model_out, monitor='val_accuracy', verbose=1, save_best_only=True) \
                   if auc_checkpointer is None else auc_checkpointer
    stdout_flush = DMFlush()
    callbacks = [early_stopping, checkpointer, stdout_flush]
    if optim == 'sgd':
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, verbose=1)
        callbacks.append(reduce_lr)

    if auto_batch_balance:
        class_weight = None
    elif nb_class == 2:
        class_weight = {0: 1.0, 1: pos_cls_weight}
    elif nb_class == 3:
        class_weight = {0: 1.0, 1: pos_cls_weight, 2: neg_cls_weight}
    else:
        class_weight = None

    pickle_safe = False  # (compat; Keras TF2 no usa pickle_safe)

    # Stage 1: top layers only
    print("Top layer nb =", top_layer_nb)
    for layer in org_model.layers[:top_layer_nb]:
        layer.trainable = False
    for layer in org_model.layers:
        try:
            if isinstance(layer, (Dense, Conv2D)) and getattr(layer, 'kernel_regularizer', None) is not None:
                layer.kernel_regularizer.l2 = weight_decay
        except AttributeError:
            pass
        if isinstance(layer, Dropout):
            layer.rate = hidden_dropout

    model.compile(optimizer=create_optimizer(optim, init_lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print("Start training on the top layers only"); sys.stdout.flush()
    hist = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        class_weight=class_weight,
        validation_data=validation_set,
        validation_steps=validation_steps,
        validation_batch_size=1,
        callbacks=callbacks,
        workers=nb_worker,
        verbose=1
    )
    print("Done.")
    try:
        loss_history = list(hist.history.get('val_loss', []))
        acc_history  = list(hist.history.get('val_accuracy', []))
    except Exception:
        loss_history, acc_history = [], []

    # Stage 2: all layers
    for layer in org_model.layers[:top_layer_nb]:
        layer.trainable = True
    for layer in org_model.layers:
        try:
            if isinstance(layer, (Dense, Conv2D)) and getattr(layer, 'kernel_regularizer', None) is not None:
                layer.kernel_regularizer.l2 = weight_decay2
        except AttributeError:
            pass
        if isinstance(layer, Dropout):
            layer.rate = hidden_dropout2

    model.compile(optimizer=create_optimizer(optim, init_lr * all_layer_multiplier),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print("Start training on all layers"); sys.stdout.flush()
    hist = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=all_layer_epochs,
        class_weight=class_weight,
        validation_data=validation_set,
        validation_steps=validation_steps,
        validation_batch_size=1,
        callbacks=callbacks,
        workers=nb_worker,
        verbose=1
    )
    print("Done.")
    try:
        loss_history.extend(hist.history.get('val_loss', []))
        acc_history.extend(hist.history.get('val_accuracy', []))
    except Exception:
        pass

    return model, np.array(loss_history), np.array(acc_history)


# ===========================================================
# Métricas (compat con original)
# ===========================================================
class DMMetrics(object):
    """Classification metrics for the DM challenge."""

    @staticmethod
    def sensitivity(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pos = K.round(K.clip(y_true, 0, 1))
        tp = K.sum(y_pos * y_pred_pos)
        pos = K.sum(y_pos)
        return tp / (pos + K.epsilon())

    @staticmethod
    def specificity(y_true, y_pred):
        y_pred_neg = 1 - K.round(K.clip(y_pred, 0, 1))
        y_neg = 1 - K.round(K.clip(y_true, 0, 1))
        tn = K.sum(y_neg * y_pred_neg)
        neg = K.sum(y_neg)
        return tn / (neg + K.epsilon())


# ===========================================================
# AUC Checkpointer (con calc_test_auc y test-time flips)
# ===========================================================
class DMAucModelCheckpoint(Callback):
    """
    Checkpointer con:
      - Mejor modelo por val_auc de Keras (logs)
      - Threshold óptimo (Youden) calculado sobre VALIDACIÓN al final
      - Guardado de:
          * metrics_final.csv  (AUC con IC95% bootstrap, Accuracy, Sensitivity, Specificity) en train/val/test
          * preds_test.csv     (y_true, y_pred_prob, y_pred_label)
          * test_confusion_matrix.png (counts y normalizada)
          * plots_train_val.png (curvas de loss/acc/AUC para TODAS las épocas; AUC train/val de logs)
    """

    def __init__(self,
                 filepath,
                 val_data,
                 test_samples=None,
                 batch_size=None,
                 test_augment=False,
                 train_data=None,
                 test_data=None,
                 out_dir=".",
                 class_names=None):
        super().__init__()
        # Entradas
        self.filepath = filepath
        self.val_data = val_data
        self.test_samples = test_samples
        self.batch_size = batch_size
        self.test_augment = test_augment
        self.train_data = train_data
        self.test_data = test_data
        self.out_dir = out_dir or "."
        os.makedirs(self.out_dir, exist_ok=True)
        self.class_names = class_names

        # Validaciones mínimas
        if isinstance(val_data, tuple):
            if batch_size is None:
                raise Exception('batch_size must be specified when validation data is loaded into RAM')
        elif test_samples is None:
            raise Exception('test_samples must be specified when val_data is a generator')

        # Estado “best”
        self.best_epoch = 0
        self.best_auc = -1.0

        # Históricos para plot final
        self.hist_train_loss = []
        self.hist_val_loss   = []
        self.hist_train_acc  = []
        self.hist_val_acc    = []
        self.hist_train_auc  = []  # AUC de entrenamiento desde logs['auc']
        self.hist_val_auc    = []  # AUC de validación desde logs['val_auc']

    # ---------- utilidades ----------
    @staticmethod
    def _augmented_predict(model, X, batch_size=None, test_augment=False):
        if test_augment:
            X_tests = flip_all_img(X)
            preds = []
            for X_t in X_tests:
                preds.append(model.predict(X_t, batch_size=batch_size, verbose=0))
            return np.stack(preds).mean(axis=0)
        return model.predict(X, batch_size=batch_size, verbose=0)

    @staticmethod
    def _collect_preds(dataset, model, batch_size=None, steps=None, test_augment=False):
        """Devuelve (y_true_onehot, y_prob) para tuple o generator."""
        if isinstance(dataset, tuple):
            X, y_true = dataset[:2]
            y_prob = DMAucModelCheckpoint._augmented_predict(model, X, batch_size, test_augment)
            return y_true, y_prob
        # generator / iterator
        if steps is None:
            try:
                steps = len(dataset)
            except Exception:
                raise Exception("steps (test_samples) requerido para iteradores.")
        try:
            dataset.reset()
        except Exception:
            pass
        ys, ps = [], []
        for i in range(steps):
            try:
                batch = dataset[i]
            except Exception:
                batch = next(dataset)
            if len(batch) > 2:
                Xb, yb, _ = batch
            else:
                Xb, yb = batch
            pb = DMAucModelCheckpoint._augmented_predict(model, Xb, batch_size, test_augment)
            ys.append(yb); ps.append(pb)
        return np.concatenate(ys), np.concatenate(ps)

    @staticmethod
    def _gmean_threshold(y_true_onehot, y_prob):
        """Umbral óptimo que maximiza G-mean = sqrt(sens * spec) (binario)."""
        if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
            p = y_prob[:, 1]; y = y_true_onehot[:, 1]
        else:
            p = y_prob.ravel(); y = y_true_onehot.ravel()
        # Usamos los mismos umbrales “candidatos” que en la versión Youden
        t_sorted = np.unique(np.round(p, 6))
        if t_sorted.size == 0: 
            return 0.5
        if t_sorted[0] > 0.0: t_sorted = np.insert(t_sorted, 0, 0.0)
        if t_sorted[-1] < 1.0: t_sorted = np.append(t_sorted, 1.0)
        best_t, best_g = 0.5, -1.0
        for t in t_sorted:
            y_hat = (p >= t).astype(np.int32)
            tp = np.sum((y == 1) & (y_hat == 1))
            fn = np.sum((y == 1) & (y_hat == 0))
            tn = np.sum((y == 0) & (y_hat == 0))
            fp = np.sum((y == 0) & (y_hat == 1))
            sens = tp / (tp + fn + 1e-9)
            spec = tn / (tn + fp + 1e-9)
            g = np.sqrt(sens * spec)
            if g > best_g:
                best_g, best_t = g, t
        return float(best_t)


    @staticmethod
    def _binary_metrics(y_true_onehot, y_prob, thr):
        if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
            p = y_prob[:, 1]; y = y_true_onehot[:, 1]
        else:
            p = y_prob.ravel(); y = y_true_onehot.ravel()
        y_hat = (p >= thr).astype(np.int32)
        tp = np.sum((y == 1) & (y_hat == 1))
        fn = np.sum((y == 1) & (y_hat == 0))
        tn = np.sum((y == 0) & (y_hat == 0))
        fp = np.sum((y == 0) & (y_hat == 1))
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, p)
        return dict(accuracy=acc, sensitivity=sens, specificity=spec, auc=auc)

    @staticmethod
    def _auc_ci_bootstrap(y_true_onehot, y_prob, n_boot=1000, seed=13):
        rng = np.random.RandomState(seed)
        if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
            p = y_prob[:, 1]; y = y_true_onehot[:, 1]
        else:
            p = y_prob.ravel(); y = y_true_onehot.ravel()
        from sklearn.metrics import roc_auc_score
        aucs = []
        n = len(y)
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            try:
                aucs.append(roc_auc_score(y[idx], p[idx]))
            except Exception:
                pass
        aucs = np.array(aucs) if len(aucs) else np.array([np.nan])
        mean = float(np.nanmean(aucs))
        lo, hi = np.nanpercentile(aucs, [2.5, 97.5])
        return mean, (float(lo), float(hi))

    def _plot_confusion_matrix(self, y_true_onehot, y_prob, thr, out_path, labels=None):
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
            p = y_prob[:, 1]; y = y_true_onehot[:, 1]
        else:
            p = y_prob.ravel(); y = y_true_onehot.ravel()
        y_hat = (p >= thr).astype(np.int32)
        cm = confusion_matrix(y, y_hat, labels=[0, 1])
        cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        classes = labels if (labels and len(labels) == 2) else ['neg', 'pos']

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]; ax.imshow(cm, interpolation='nearest')
        ax.set_title('Confusion Matrix (counts)')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(classes); ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center")

        ax = axes[1]; ax.imshow(cmn, interpolation='nearest')
        ax.set_title('Confusion Matrix (row-normalized)')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(classes); ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center")

        fig.tight_layout()
        fig.savefig(out_path, dpi=160); plt.close(fig)
        print(f">>> Guardado: {out_path}")

    # ---------- Keras hooks ----------
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Guardar históricos desde los logs de Keras
        self.hist_train_loss.append(float(logs.get('loss', float('nan'))))
        self.hist_val_loss.append(float(logs.get('val_loss', float('nan'))))
        self.hist_train_acc.append(float(logs.get('accuracy', float('nan'))))
        self.hist_val_acc.append(float(logs.get('val_accuracy', float('nan'))))
        # AUC desde logs (sin recalcular manualmente)
        self.hist_train_auc.append(float(logs.get('auc', float('nan'))))
        self.hist_val_auc.append(float(logs.get('val_auc', float('nan'))))

        # Selección de mejor modelo por val_auc de Keras
        val_auc_now = self.hist_val_auc[-1]
        print(f" - Epoch:{epoch + 1}, val_auc(logs)={val_auc_now:.4f}")
        if not np.isnan(val_auc_now) and val_auc_now > self.best_auc:
            self.best_epoch = epoch + 1
            self.best_auc = val_auc_now
            if self.filepath != "NOSAVE":
                try:
                    self.model.save(self.filepath)
                except Exception as e:
                    print(f"[DMAucModelCheckpoint] No se pudo guardar el mejor modelo: {e}")

    def on_train_end(self, logs=None):
        print(f"\n>>> Best val AUC (logs): {self.best_auc:.4f} @ epoch {self.best_epoch}. Guardando artefactos…")
        # 1) Threshold óptimo con VALIDACIÓN (predicción al final)
        yv, pv = self._collect_preds(self.val_data, self.model,
                                     batch_size=self.batch_size,
                                     steps=self.test_samples,
                                     test_augment=self.test_augment)
        thr = self._gmean_threshold(yv, pv)

        print(f">>> Threshold óptimo (G-mean) = {thr:.4f}")


        # 2) Métricas en TRAIN / VAL / TEST
        def eval_split(dataset, steps=None):
            y, p = self._collect_preds(dataset, self.model,
                                       batch_size=self.batch_size,
                                       steps=steps, test_augment=False)
            metr = self._binary_metrics(y, p, thr)
            auc_mean, (auc_lo, auc_hi) = self._auc_ci_bootstrap(y, p)
            metr.update(dict(auc_ci_lo=auc_lo, auc_ci_hi=auc_hi))
            return metr, y, p

        rows = []
        if self.train_data is not None:
            mt, _, _ = eval_split(self.train_data,
                                  steps=(len(self.train_data) if not isinstance(self.train_data, tuple) else None))
            rows.append(("train", mt))

        mv, _, _ = eval_split(self.val_data, steps=self.test_samples)
        rows.append(("val", mv))

        preds_csv_path = None
        y_test = p_test = None
        if self.test_data is not None:
            steps_test = None if isinstance(self.test_data, tuple) else len(self.test_data)
            mtst, y_test, p_test = eval_split(self.test_data, steps=steps_test)
            rows.append(("test", mtst))
            # CSV test
            if p_test.ndim == 2 and p_test.shape[1] >= 2:
                prob = p_test[:, 1]; y_true = y_test[:, 1]
            else:
                prob = p_test.ravel(); y_true = y_test.ravel()
            y_hat = (prob >= thr).astype(np.int32)
            preds_csv_path = os.path.join(self.out_dir, "preds_test.csv")
            import csv
            with open(preds_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["y_true", "y_pred_prob", "y_pred_label"])
                for yt, pr, yh in zip(y_true, prob, y_hat):
                    w.writerow([int(yt), float(pr), int(yh)])
            print(f">>> Guardado: {preds_csv_path}")
            # Matriz de confusión
            cm_path = os.path.join(self.out_dir, "test_confusion_matrix.png")
            self._plot_confusion_matrix(y_test, p_test, thr, cm_path, labels=self.class_names)

        # CSV de métricas finales
        metrics_csv = os.path.join(self.out_dir, "metrics_final.csv")
        import csv
        with open(metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split","AUC","AUC_CI95_lo","AUC_CI95_hi","Accuracy","Sensitivity","Specificity","Threshold"])
            for split, m in rows:
                w.writerow([
                    split,
                    f"{m['auc']:.6f}",
                    f"{m['auc_ci_lo']:.6f}",
                    f"{m['auc_ci_hi']:.6f}",
                    f"{m['accuracy']:.6f}",
                    f"{m['sensitivity']:.6f}",
                    f"{m['specificity']:.6f}",
                    f"{thr:.6f}",
                ])
        print(f">>> Guardado: {metrics_csv}")

        # 3) Plot agregado final (loss/acc/AUC train y val — desde logs)
        try:
            import matplotlib.pyplot as plt
            ep = np.arange(1, len(self.hist_train_loss)+1)
            fig, ax = plt.subplots(3, 1, figsize=(8, 12))
            ax[0].plot(ep, self.hist_train_loss, label="train_loss")
            ax[0].plot(ep, self.hist_val_loss,   label="val_loss")
            ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()

            ax[1].plot(ep, self.hist_train_acc, label="train_acc")
            ax[1].plot(ep, self.hist_val_acc,   label="val_acc")
            ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()

            ax[2].plot(ep, self.hist_train_auc, label="train_auc")
            ax[2].plot(ep, self.hist_val_auc,   label="val_auc")
            ax[2].set_title("AUC"); ax[2].set_xlabel("epoch"); ax[2].legend()

            fig.tight_layout()
            plot_path = os.path.join(self.out_dir, "plots_train_val.png")
            fig.savefig(plot_path, dpi=160); plt.close(fig)
            print(f">>> Guardado: {plot_path}")
        except Exception as e:
            print("No se pudo generar plots:", e)

        print(f">>> Modelo con mejor AUC guardado en: {self.filepath if self.filepath!='NOSAVE' else '(no guardado)'}")
        sys.stdout.flush()

# ===========================================================
# Callback utilitario: flush stdout
# ===========================================================
class DMFlush(Callback):
    """Flush de stdout al inicio/fin de cada época."""
    def __init__(self):
        super(DMFlush, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        sys.stdout.flush()
