#!/usr/bin/env python3
import os, argparse, sys, warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from dm_image import DMImageDataGenerator

from dm_keras_ext import do_3stage_training, DMAucModelCheckpoint
from contextlib import nullcontext

from dm_resnet import add_top_layers, bottleneck_org

warnings.filterwarnings('ignore', category=UserWarning)
# …tus demás imports…

# --- Best-epoch logger por val_loss ---
class BestByValLoss(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_epoch = None
        self.best_val = float("inf")
        self.best_logs = None
        self.stage = 0   # por si hay varias .fit (etapas)

    def on_train_begin(self, logs=None):
        # etiqueta de etapa (opcional)
        self.stage += 1

    def on_epoch_end(self, epoch, logs=None):
        if logs is None: 
            return
        v = logs.get("val_loss")
        if v is None:
            return
        if v < self.best_val:
            self.best_val = v
            # guardamos una copia plana de las métricas de esa época
            self.best_epoch = epoch
            self.best_logs = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
            # también nos llevamos loss/acc/auc de train explícitamente
            for k in ("loss","accuracy","auc"):
                if k in logs:
                    self.best_logs[k] = float(logs[k])

    def summary_row(self):
        """Devuelve un dict listo para CSV con métricas de train/val en la mejor época."""
        d = dict(
            best_epoch=int(self.best_epoch) if self.best_epoch is not None else -1,
            val_loss=self.best_logs.get("val_loss", float("nan")) if self.best_logs else float("nan"),
            val_accuracy=self.best_logs.get("val_accuracy", float("nan")) if self.best_logs else float("nan"),
            val_auc=self.best_logs.get("val_auc", float("nan")) if self.best_logs else float("nan"),
            train_loss=self.best_logs.get("loss", float("nan")) if self.best_logs else float("nan"),
            train_accuracy=self.best_logs.get("accuracy", float("nan")) if self.best_logs else float("nan"),
            train_auc=self.best_logs.get("auc", float("nan")) if self.best_logs else float("nan"),
        )
        return d



def run(train_dir, val_dir, test_dir, patch_model_state=None, resume_from=None,
        img_size=[1152, 896], img_scale=None, rescale_factor=None,
        featurewise_center=True, featurewise_mean=60.2,
        equalize_hist=False, augmentation=True,
        class_list=['neg', 'pos'], patch_net='resnet50',
        block_type='resnet', top_depths=[512, 512], top_repetitions=[3, 3],
        bottleneck_enlarge_factor=4,
        add_heatmap=False, avg_pool_size=[7, 7],
        add_conv=True, add_shortcut=False,
        hm_strides=(1,1), hm_pool_size=(5,5),
        fc_init_units=64, fc_layers=2,
        top_layer_nb=None,
        batch_size=4, train_bs_multiplier=.5,
        nb_epoch=0, all_layer_epochs=50,
        load_val_ram=False, load_train_ram=False,
        weight_decay=.001, hidden_dropout=.0,
        weight_decay2=.01, hidden_dropout2=.0,
        optim='adam', init_lr=.0001, lr_patience=2, es_patience=10,
        all_layer_multiplier=.01,
        best_model="./modelState/image_clf.h5", final_model="NOSAVE",
        multi_gpu=False,
        # ↓↓↓ añadidos ↓↓↓
        auto_batch_balance=True, pos_cls_weight=1.0, neg_cls_weight=3.5,
        auto_level=False, auto_level_clip=0.0, top_layer_epochs=50, 
        top_layer_multiplier=0.1,
    ):


    """
    Entrena un clasificador de imágenes imitando la receta original de los autores (INbreast).
    - Tamaño 1152x896
    - Sin img_scale (no normaliza por target_scale)
    - rescale_factor ~ 255/65535
    - featurewise_center con mean fija 44.33 (escala 0..255)
    - batch_size=4, Adam 1e-4, wd=1e-3/1e-2, nb_epoch=0, all_layer_epochs=50
    """

    # ====== Sanity checks / paths ====== #
    if not os.path.isdir(train_dir):
        raise ValueError("train_dir no existe: %s" % train_dir)
    if not os.path.isdir(val_dir):
        raise ValueError("val_dir no existe: %s" % val_dir)
    if not os.path.isdir(test_dir):
        raise ValueError("test_dir no existe: %s" % test_dir)

    # (modelo se construye dentro de strategy.scope() más abajo)

    # ====== Definición de tamaño/canales ====== #
    # Modelo espera (H,W)
    expected_h, expected_w = img_size[0], img_size[1]
    target_hw = (expected_h, expected_w)

    # color_mode según backbone
    if patch_net != 'yaroslav':
        color_mode = "rgb"         # VGG/ResNet (3 canales)
    else:
        color_mode = "grayscale"   # otros que acepten 1 canal

    # --- Preprocessing que emula lo necesario del pipeline original ---
    def _preproc_fn(x):
        """
        Pipeline por imagen:
          1) (opcional) auto-level por canal (tipo ImageMagick) con clip en percentil
          2) rescale_factor
          3) centering por media fija
          4) si pedimos RGB y llega 1 canal, duplicar
        """
        # 1) AUTO-LEVEL (lineal por canal). Si auto_level_clip>0, usa percentiles [p, 100-p]
        if auto_level:
            x_float = x.astype(np.float32, copy=False)
            # calcular low/high por canal manteniendo dims para broadcasting
            if x_float.ndim == 3 and x_float.shape[-1] in (1, 3):
                axes = (0, 1)
                keep = True
                if auto_level_clip and auto_level_clip > 0.0:
                    low = np.percentile(x_float, auto_level_clip, axis=axes, keepdims=keep)
                    high = np.percentile(x_float, 100.0 - auto_level_clip, axis=axes, keepdims=keep)
                else:
                    low = x_float.min(axis=axes, keepdims=keep)
                    high = x_float.max(axis=axes, keepdims=keep)
                scale = 255.0 / np.maximum(high - low, 1e-5)
                x_float = (x_float - low) * scale
                x_float = np.clip(x_float, 0.0, 255.0)
                x = x_float
            else:
                # caso raro: sin canal de color explícito
                if auto_level_clip and auto_level_clip > 0.0:
                    low = np.percentile(x, auto_level_clip)
                    high = np.percentile(x, 100.0 - auto_level_clip)
                else:
                    low = float(np.min(x))
                    high = float(np.max(x))
                scale = 255.0 / max(high - low, 1e-5)
                x = (x.astype(np.float32) - low) * scale
                x = np.clip(x, 0.0, 255.0)

        # 2) RESCALE
        if rescale_factor is not None:
            x = x * rescale_factor

        # 3) CENTERING por media fija
        if featurewise_center and (featurewise_mean is not None):
            x = x - featurewise_mean

        # 4) asegurar 3 canales si el backbone lo pide
        if color_mode == "rgb" and x.ndim == 3 and x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)

        return x


    # Generadores del repo (respetan el pipeline DM)
    train_datagen = DMImageDataGenerator()
    val_datagen   = DMImageDataGenerator()
    test_datagen  = DMImageDataGenerator()

    # Los backbones ImageNet requieren 3 canales; Yaroslav puede 1 canal
    dup_rgb = (patch_net != 'yaroslav')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(expected_h, expected_w),   # (H, W)
        target_scale=img_scale,                 # None si --no-img-scale
        gs_255=False,                           # lectura 16-bit/UNCHANGED como en DM
        equalize_hist=equalize_hist,
        rescale_factor=rescale_factor,          # p. ej. 255/65535
        dup_3_channels=dup_rgb,                 # 3 canales si backbone ImageNet
        data_format='default',
        classes=class_list,
        class_mode='categorical',
        auto_batch_balance=False,               # el repo deja class_weight si aplica
        batch_size=batch_size,
        preprocess=_preproc_fn if callable(_preproc_fn) else None,
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(expected_h, expected_w),
        target_scale=img_scale,
        gs_255=False,
        equalize_hist=equalize_hist,
        rescale_factor=rescale_factor,
        dup_3_channels=dup_rgb,
        data_format='default',
        classes=class_list,
        class_mode='categorical',
        auto_batch_balance=False,
        batch_size=batch_size,
        preprocess=_preproc_fn if callable(_preproc_fn) else None,
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(expected_h, expected_w),
        target_scale=img_scale,
        gs_255=False,
        equalize_hist=equalize_hist,
        rescale_factor=rescale_factor,
        dup_3_channels=dup_rgb,
        data_format='default',
        classes=class_list,
        class_mode='categorical',
        auto_batch_balance=False,
        batch_size=batch_size,
        preprocess=_preproc_fn if callable(_preproc_fn) else None,
        shuffle=False
    )


    # ====== Carga a RAM opcional ====== #
    # Para emular --load-train-ram y --load-val-ram del script
    def _load_to_ram(gen, max_batches=None):
        xs, ys = [], []
        # empezar siempre desde el inicio si es posible
        try:
            gen.reset()
        except Exception:
            pass
        steps = len(gen) if max_batches is None else min(len(gen), max_batches)
        for _ in range(steps):
            batch = next(gen)          # clave: usar next(gen), NO gen[i]
            if isinstance(batch, (list, tuple)):
                x, y = batch[:2]
            else:
                x, y = batch, None
            xs.append(x)
            ys.append(y)
        X = np.concatenate(xs, axis=0) if xs else None
        Y = np.concatenate([y for y in ys if y is not None], axis=0) if any(y is not None for y in ys) else None
        return (X, Y)

    def _as_yielding(gen):
        while True:
            yield next(gen)




    train_set = train_generator
    validation_set = validation_generator
    test_set = test_generator

    if load_train_ram:
        print(">> Cargando TRAIN en RAM ...")
        Xtr, Ytr = _load_to_ram(train_generator)
        train_set = (Xtr, Ytr)

    if load_val_ram:
        print(">> Cargando VAL en RAM ...")
        Xv, Yv = _load_to_ram(validation_generator)
        validation_set = (Xv, Yv)
        val_samples = len(Xv) if isinstance(validation_set, tuple) else None
    else:
        val_samples = len(validation_generator)

    train_batches = len(train_generator)
    validation_steps = None if isinstance(validation_set, tuple) else len(validation_generator)
    nb_worker = 1  # para reproducibilidad
    # después de calcular train_batches / validation_steps ...
    train_input = train_set if isinstance(train_set, tuple) else _as_yielding(train_generator)
    val_input   = validation_set if isinstance(validation_set, tuple) else _as_yielding(validation_generator)

    # ====== Multi-GPU (estrategia TF) ====== #
    if multi_gpu:
        try:
            strategy = tf.distribute.MirroredStrategy()
            print(">> Usando MirroredStrategy en GPUs:", strategy.num_replicas_in_sync)
        except:
            from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
            strategy = MirroredStrategy()
            print(">> Usando MirroredStrategy (fallback)")
    else:
        class DummyStrategy:
            def scope(self):
                from contextlib import contextmanager
                @contextmanager
                def cm():
                    yield
                return cm()
        strategy = DummyStrategy()

    # ====== Compilar y entrenar (TODO dentro del scope) ====== #
    with strategy.scope():
        # === Construcción/recuperación del modelo (DENTRO del scope) ===
        if resume_from is not None:
            print(">> Cargando modelo completo desde:", resume_from)
            image_model = load_model(resume_from, compile=False)
            org_model = image_model  # compat
            # top_layer_nb: usa el que venga por args (p.ej., 2) o None
        else:
            if patch_model_state is None:
                raise ValueError("--resume-from o --patch-model-state es requerido")
            print(">> Cargando patch model (base) desde:", patch_model_state)
            patch_model = load_model(patch_model_state, compile=False)
            print(">> Añadiendo top layers...")
            image_model, top_layer_nb = add_top_layers(
                patch_model,
                net=patch_net,
                block_type=block_type,
                top_depths=top_depths,
                top_repetitions=top_repetitions,
                bottleneck_enlarge_factor=bottleneck_enlarge_factor,
                add_heatmap=add_heatmap,
                avg_pool_size=avg_pool_size,
                add_conv=add_conv,
                add_shortcut=add_shortcut,
                hm_strides=hm_strides,
                hm_pool_size=hm_pool_size,
                fc_init_units=fc_init_units,
                fc_layers=fc_layers,
                nb_class=len(class_list),
                shortcut_with_bn=True,
                weight_decay=weight_decay
            )
            org_model = image_model.layers[1] if len(image_model.layers) > 1 else image_model

                # === AUC checkpointer según tipo de validación (con extras) ===
                # === AUC checkpointer según tipo de validación (con extras) ===
        if isinstance(validation_set, tuple):
            batch_for_tuple = batch_size
            val_steps = None
        else:
            batch_for_tuple = None
            val_steps = validation_steps if validation_steps is not None else len(validation_set)

        auc_checkpointer = DMAucModelCheckpoint(
            best_model,
            val_data=validation_set,                 # <--- nombre nuevo
            test_samples=val_steps,
            batch_size=batch_for_tuple,
            test_augment=False,
            train_data=train_set if 'train_set' in locals() else None,
            test_data=test_set if 'test_set' in locals() else test_generator,
            out_dir=os.path.dirname(best_model) or ".",
            class_names=class_list
        )


        print("\n>> Compilando y entrenando (3-stage fine-tuning)...")

        # 1) Callback para capturar la mejor época por val_loss
        best_cb = BestByValLoss()   # (definido en image_clf_train.py como te pasé)      

        # 2) Llamada a do_3stage_training con extra_callbacks y multiplicadores correctos
        image_model, loss_hist, acc_hist = do_3stage_training(
            image_model, org_model,
            train_generator=train_input,            # wrapper/iterator
            validation_set=val_input,
            validation_steps=validation_steps,      # steps para val si NO es (X,y)
            best_model_out=best_model,
            steps_per_epoch=train_batches,
            top_layer_nb=top_layer_nb,              # p.ej. 11 para VGG16
            nb_epoch=nb_epoch,                      # etapa 1
            top_layer_epochs=top_layer_epochs,      # etapa 2 (p.ej. 15)
            all_layer_epochs=all_layer_epochs,      # etapa 3 (p.ej. 100)
            optim=optim,
            init_lr=init_lr,
            top_layer_multiplier=top_layer_multiplier,   # ← usa el de “top” (recomendado 0.1)
            all_layer_multiplier=all_layer_multiplier,   # ← usa el de “all” (recomendado 0.02)
            es_patience=es_patience,
            lr_patience=lr_patience,
            auto_batch_balance=auto_batch_balance,
            nb_class=len(class_list),
            pos_cls_weight=pos_cls_weight,          # en general 1
            neg_cls_weight=neg_cls_weight,          # en general 1
            nb_worker=1,
            auc_checkpointer=auc_checkpointer,
            test_generator=test_generator,
            extra_callbacks=[best_cb]               # ← NUEVO
        )

        # 3) AHORA sí, escribir el CSV con la mejor época por val_loss
        row = best_cb.summary_row()
        out_csv = os.path.join(os.path.dirname(best_model), "metrics_fit_best_epoch.csv")

        import csv, math
        with open(out_csv, "w", newline="") as f:
            fieldnames = ["best_epoch","val_loss","val_accuracy","val_auc",
                        "train_loss","train_accuracy","train_auc"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            # Asegura que existan todas las llaves aunque alguna métrica no esté
            safe_row = {k: (row.get(k) if row and (k in row) else float("nan")) for k in fieldnames}
            w.writerow(safe_row)

        print(">> Guardado:", out_csv)
        print("   Best epoch:", safe_row["best_epoch"],
            "| val_loss:", safe_row["val_loss"],
            "| val_auc:", safe_row["val_auc"],
            "| val_acc:", safe_row["val_accuracy"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DM image clf training")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("val_dir", type=str)
    parser.add_argument("test_dir", type=str)

    parser.add_argument("--patch-model-state", dest="patch_model_state", type=str, default=None)
    parser.add_argument("--no-patch-model-state", dest="patch_model_state", action="store_const", const=None)
    parser.add_argument("--resume-from", dest="resume_from", type=str, default=None)
    parser.add_argument("--no-resume-from", dest="resume_from", action="store_const", const=None)

    parser.add_argument("--img-size", "-is", dest="img_size", nargs=2, type=int, default=[1152, 896])
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=float, default=None)
    parser.add_argument("--no-img-scale", "-nic", dest="img_scale", action="store_const", const=None)
    parser.add_argument("--rescale-factor", dest="rescale_factor", type=float, default=0.003891)
    parser.add_argument("--no-rescale-factor", dest="rescale_factor", action="store_const", const=None)

    parser.add_argument("--featurewise-center", dest="featurewise_center", action="store_true")
    parser.add_argument("--no-featurewise-center", dest="featurewise_center", action="store_false")
    parser.set_defaults(featurewise_center=True)
    parser.add_argument("--featurewise-mean", dest="featurewise_mean", type=float, default=44.33)

    parser.add_argument("--auto-level", dest="auto_level", action="store_true", help="Activa el autolevel en preprocesamiento")
    parser.add_argument("--no-auto-level", dest="auto_level", action="store_false")
    parser.set_defaults(auto_level=False)
    parser.add_argument("--auto-level-clip", dest="auto_level_clip", type=float, default=0.0,
                        help="Percentil de recorte para autolevel (0.0 = sin recorte)")

    parser.add_argument("--equalize-hist", dest="equalize_hist", action="store_true")
    parser.add_argument("--no-equalize-hist", dest="equalize_hist", action="store_false")
    parser.set_defaults(equalize_hist=False)

    parser.add_argument("--batch-size", "-bs", dest="batch_size", type=int, default=4)
    parser.add_argument("--train-bs-multiplier", dest="train_bs_multiplier", type=float, default=.5)
    parser.add_argument("--augmentation", dest="augmentation", action="store_true")
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.set_defaults(augmentation=True)
    parser.add_argument("--class-list", dest="class_list", nargs='+', type=str, default=['neg', 'pos'])

    parser.add_argument("--patch-net", dest="patch_net", type=str, default="resnet50")
    parser.add_argument("--block-type", dest="block_type", type=str, default="resnet")
    parser.add_argument("--top-depths", dest="top_depths", nargs='+', type=int, default=[512, 512])
    parser.add_argument("--top-repetitions", dest="top_repetitions", nargs='+', type=int, default=[3, 3])
    parser.add_argument("--bottleneck-enlarge-factor", dest="bottleneck_enlarge_factor", type=int, default=4)
    parser.add_argument("--add-heatmap", dest="add_heatmap", action="store_true")
    parser.add_argument("--no-add-heatmap", dest="add_heatmap", action="store_false")
    parser.set_defaults(add_heatmap=False)
    parser.add_argument("--avg-pool-size", dest="avg_pool_size", nargs=2, type=int, default=[7, 7])
    parser.add_argument("--add-conv", dest="add_conv", action="store_true")
    parser.add_argument("--no-add-conv", dest="add_conv", action="store_false")
    parser.set_defaults(add_conv=True)
    parser.add_argument("--add-shortcut", dest="add_shortcut", action="store_true")
    parser.add_argument("--no-add-shortcut", dest="add_shortcut", action="store_false")
    parser.set_defaults(add_shortcut=False)
    parser.add_argument("--hm-strides", dest="hm_strides", nargs=2, type=int, default=[1, 1])
    parser.add_argument("--hm-pool-size", dest="hm_pool_size", nargs=2, type=int, default=[5, 5])
    parser.add_argument("--fc-init-units", dest="fc_init_units", type=int, default=64)
    parser.add_argument("--fc-layers", dest="fc_layers", type=int, default=2)

    # === Cambios clave ===
    parser.add_argument("--top-layer-nb", dest="top_layer_nb", type=int, default=11)          # antes None
    parser.add_argument("--no-top-layer-nb", dest="top_layer_nb", action="store_const", const=None)
    parser.add_argument("--nb-epoch", "-ne", dest="nb_epoch", type=int, default=10)            # etapa 1 (antes 0)
    parser.add_argument("--top-layer-epochs", dest="top_layer_epochs", type=int, default=15)   # etapa 2 (antes 0)
    parser.add_argument("--all-layer-epochs", dest="all_layer_epochs", type=int, default=100)  # etapa 3 (antes 50)

    parser.add_argument("--load-val-ram", dest="load_val_ram", action="store_true")
    parser.add_argument("--no-load-val-ram", dest="load_val_ram", action="store_false")
    parser.set_defaults(load_val_ram=True)
    parser.add_argument("--load-train-ram", dest="load_train_ram", action="store_true")
    parser.add_argument("--no-load-train-ram", dest="load_train_ram", action="store_false")
    parser.set_defaults(load_train_ram=True)

    parser.add_argument("--weight-decay", "-wd", dest="weight_decay", type=float, default=.0001)
    parser.add_argument("--hidden-dropout", "-hd", dest="hidden_dropout", type=float, default=.2)
    parser.add_argument("--weight-decay2", "-wd2", dest="weight_decay2", type=float, default=.0001)
    parser.add_argument("--hidden-dropout2", "-hd2", dest="hidden_dropout2", type=float, default=.3)

    parser.add_argument("--optimizer", dest="optim", type=str, default="adam")
    parser.add_argument("--init-learningrate", "-ilr", dest="init_lr", type=float, default=.001)
    parser.add_argument("--lr-patience", "-lrp", dest="lr_patience", type=int, default=4)
    parser.add_argument("--es-patience", "-esp", dest="es_patience", type=int, default=8)

    parser.add_argument("--auto-batch-balance", dest="auto_batch_balance", action="store_true")
    parser.add_argument("--no-auto-batch-balance", dest="auto_batch_balance", action="store_false")
    parser.set_defaults(auto_batch_balance=True)
    parser.add_argument("--pos-cls-weight", dest="pos_cls_weight", type=float, default=1.0)
    parser.add_argument("--neg-cls-weight", dest="neg_cls_weight", type=float, default=1.0)

    # === NUEVO: multiplicadores separados
    parser.add_argument("--top-layer-multiplier", dest="top_layer_multiplier", type=float, default=0.1)
    parser.add_argument("--all-layer-multiplier", dest="all_layer_multiplier", type=float, default=.02)

    parser.add_argument("--best-model", "-bm", dest="best_model", type=str, default="./modelState/image_clf.h5")
    parser.add_argument("--final-model", "-fm", dest="final_model", type=str, default="NOSAVE")
    parser.add_argument("--multi-gpu", action="store_true", help="Usar tf.distribute.MirroredStrategy con las GPUs visibles")

    args = parser.parse_args()

    run_opts = dict(
        patch_model_state=args.patch_model_state,
        resume_from=args.resume_from,
        img_size=args.img_size,
        img_scale=args.img_scale,
        rescale_factor=args.rescale_factor,
        featurewise_center=args.featurewise_center,
        featurewise_mean=args.featurewise_mean,
        equalize_hist=args.equalize_hist,
        batch_size=args.batch_size,
        train_bs_multiplier=args.train_bs_multiplier,
        augmentation=args.augmentation,
        class_list=args.class_list,
        patch_net=args.patch_net,
        block_type=args.block_type,
        top_depths=args.top_depths,
        top_repetitions=args.top_repetitions,
        bottleneck_enlarge_factor=args.bottleneck_enlarge_factor,
        add_heatmap=args.add_heatmap,
        avg_pool_size=args.avg_pool_size,
        add_conv=args.add_conv,
        add_shortcut=args.add_shortcut,
        hm_strides=args.hm_strides,
        hm_pool_size=args.hm_pool_size,
        fc_init_units=args.fc_init_units,
        fc_layers=args.fc_layers,

        # === Cambios clave ===
        top_layer_nb=args.top_layer_nb,
        nb_epoch=args.nb_epoch,
        top_layer_epochs=args.top_layer_epochs,
        all_layer_epochs=args.all_layer_epochs,

        load_val_ram=args.load_val_ram,
        load_train_ram=args.load_train_ram,
        weight_decay=args.weight_decay,
        hidden_dropout=args.hidden_dropout,
        weight_decay2=args.weight_decay2,
        hidden_dropout2=args.hidden_dropout2,

        optim=args.optim,
        init_lr=args.init_lr,
        lr_patience=args.lr_patience,
        es_patience=args.es_patience,
        auto_batch_balance=args.auto_batch_balance,
        pos_cls_weight=args.pos_cls_weight,
        neg_cls_weight=args.neg_cls_weight,

        # === NUEVO: multiplicadores separados
        top_layer_multiplier=args.top_layer_multiplier,
        all_layer_multiplier=args.all_layer_multiplier,

        best_model=args.best_model,
        final_model=args.final_model,
        multi_gpu=args.multi_gpu,
        auto_level=args.auto_level,
        auto_level_clip=args.auto_level_clip,
    )

    print("\ntrain_dir=%s" % (args.train_dir))
    print("val_dir=%s" % (args.val_dir))
    print("test_dir=%s" % (args.test_dir))
    print("\n>>> Model training options: <<<\n", run_opts, "\n")
    run(args.train_dir, args.val_dir, args.test_dir, **run_opts)
