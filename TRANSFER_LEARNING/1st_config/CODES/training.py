#!/usr/bin/env python3
# transfer_learning_finetune_fixed.py
# Selección de GPUs desde CLI/ENV + diagnósticos y (opcional) entrenamiento

import os, sys, time, argparse, subprocess

def parse_args():
    p = argparse.ArgumentParser(
        description="Configura GPUs y lanza diagnósticos antes del entrenamiento."
    )
    p.add_argument("--gpus", type=str, default=None,
                   help="Lista de GPUs visibles, p.ej. '0,1'. Si se omite, usa CUDA_VISIBLE_DEVICES o todas.")
    p.add_argument("--threads", type=int, default=4, help="Hilos CPU para TF (intra/inter).")
    p.add_argument("--growth", action="store_true", default=True, help="VRAM dinámica (ON por defecto).")
    p.add_argument("--no-growth", dest="growth", action="store_false", help="Desactivar VRAM dinámica.")
    p.add_argument("--cuda-bin", type=str, default=None,
                   help="Binarios CUDA para anteponer al PATH (p.ej. /usr/local/cuda-11.2/bin).")
    p.add_argument("--cuda-lib", type=str, default=None,
                   help="Libs CUDA para anteponer a LD_LIBRARY_PATH (p.ej. /usr/local/cuda-11.2/lib64).")
    p.add_argument("--matmul-size", type=int, default=2048, help="Tamaño del matmul de prueba.")
    p.add_argument("--diagnostics-only", action="store_true",
                   help="Solo diagnósticos GPU/TF, no lanzar entrenamiento.")
    return p.parse_args()

def show_nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
            stderr=subprocess.STDOUT
        ).decode()
        print("== nvidia-smi (procesos de cómputo) ==")
        print(out.strip() or "(sin procesos de cómputo)")
        if str(os.getpid()) in out:
            print("✅ Este PID aparece en nvidia-smi (usando GPU).")
        else:
            print("⚠ Este PID aún no aparece en nvidia-smi.")
    except Exception as e:
        print("nvidia-smi no disponible o sin permisos:", e)

def run_training():
    # Comando sin class_weight; dejamos auto-batch-balance por defecto (ON)
    cmd = [
        sys.executable, "image_clf_train.py",
        "--multi-gpu",
        "--no-patch-model-state",
        "--resume-from", "inbreast_vgg16_512x1.h5",

        "--img-size", "1152", "896",
        "--no-img-scale",
        "--rescale-factor", "0.003891",
        "--featurewise-center",
        "--featurewise-mean", "60.20",
        "--no-equalize-hist",

        # ⇣ 
        "--batch-size", "16",
        "--train-bs-multiplier", "0.5",
        "--augmentation",
        "--class-list", "neg", "pos",

        # Sin sesgo adicional
        "--neg-cls-weight", "1",
        "--pos-cls-weight", "1",
        "--nb-epoch", "0",
        # === ETAPA 1 (solo top) ===
        "--top-layer-epochs", "50",
        "--top-layer-multiplier", "1",
        "--top-layer-nb", "3",##Entrenamiento desde el primer bloque conv.

        # === ETAPA 2 (toda la red) ===
        "--all-layer-epochs", "30",
        "--all-layer-multiplier", "0.001",

        "--optimizer", "adam",
        "--weight-decay", "0.0001",
        "--hidden-dropout", "0.2",
        "--weight-decay2", "0.0015",
        "--hidden-dropout2", "0.5",
        "--init-learningrate", "0.0002",
        "--es-patience", "3",  # un poco más estricto
        "--lr-patience", "2",

        "--no-load-train-ram",
        "--no-load-val-ram",

        "--best-model", "best_model_final_all_epoch.h5",
        "--final-model", "NOSAVE",
        ## Path to data dirs (deben contener subdirs TRAIN, VALIDATION, TEST), dentro de cada carpeta las subcarpetas por clase (neg, pos)
        "Images/TRAIN",
        "Images/VALIDATION",
        "Images/TEST"
    ]


    print("\n=== Ejecutando entrenamiento (image_clf_train.py) ===\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    print("\n=== Entrenamiento finalizado con código:", proc.returncode, "===\n")

def main():
    args = parse_args()

    # Logs TF (0=INFO,1=WARNING,2=ERROR)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    # Selección de GPUs (CLI > ENV > todas)
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # VRAM dinámica
    if args.growth:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # PATH CUDA opcional
    if args.cuda_bin:
        os.environ["PATH"] = args.cuda_bin + ":" + os.environ.get("PATH", "")

    # LD_LIBRARY_PATH: prioriza $CONDA_PREFIX/lib y luego --cuda-lib si se pasó
    conda_prefix = os.environ.get("CONDA_PREFIX")
    lib_paths = []
    if conda_prefix:
        lib_paths.append(os.path.join(conda_prefix, "lib"))
    if args.cuda_lib:
        lib_paths.insert(0, args.cuda_lib)
    if lib_paths:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join([p for p in lib_paths if p]) + (":" + current_ld if current_ld else "")

    # NCCL
    os.environ.setdefault("NCCL_LAUNCH_MODE", "GROUP")

    print("== ENV (antes de importar TensorFlow) ==")
    print("PID:", os.getpid())
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("PATH (head):", os.environ.get("PATH", "")[:180] + "...")
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))

    # Importar TensorFlow DESPUÉS de configurar el entorno
    import tensorflow as tf

    print("TF __version__:", tf.__version__)
    try:
        build = tf.sysconfig.get_build_info()
        print("TF build CUDA:", build.get("cuda_version"))
        print("TF build cuDNN:", build.get("cudnn_version"))
    except Exception as e:
        print("No se pudo leer build_info:", e)

    # Hilos CPU
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    # GPUs y growth por dispositivo
    phys = tf.config.list_physical_devices('GPU')
    for g in phys:
        try:
            tf.config.experimental.set_memory_growth(g, args.growth)
        except Exception as e:
            print("Warning set_memory_growth:", e)
    logi = tf.config.list_logical_devices('GPU')

    print("== DEVICES ==")
    print("Physical GPUs:", phys)
    print("Logical  GPUs:", logi)

    # Prueba de colocación
    print("== PLACEMENT CHECK ==")
    tf.debugging.set_log_device_placement(True)
    sz = args.matmul_size
    a = tf.random.normal([sz, sz])
    b = tf.random.normal([sz, sz])
    t0 = time.time()
    with tf.device('/GPU:0' if logi else '/CPU:0'):
        c = tf.linalg.matmul(a, b)
    _ = c.numpy()
    t1 = time.time()
    tf.debugging.set_log_device_placement(False)
    print(f"MatMul tiempo: {t1 - t0:.3f}s  (dispositivo esperado: {'GPU' if logi else 'CPU'})")

    # Diagnóstico de procesos GPU
    show_nvidia_smi()
    print("====== FIN GPU DIAGNOSTICS ======\n")

    # Entrenamiento (a menos que pidas solo diagnósticos)
    if not args.diagnostics_only:
        run_training()

if __name__ == "__main__":
    main()
