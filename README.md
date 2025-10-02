# Breast Cancer Detection System

## Description

This repository implements a comprehensive breast cancer detection system combining deep learning techniques and radiomics analysis. The system is designed for binary classification of mammographic images to identify malignant lesions.

### Key Components

- **Transfer Learning**: Fine-tuning pre-trained convolutional neural networks (VGG16) on medical imaging datasets
- **Inference Pipeline**: Batch prediction and evaluation on multiple test datasets
- **Radiomic Analysis**: Feature extraction from images and machine learning classification using traditional ML algorithms

### Supported Datasets

- CBIS-DDSM (Curated Breast Imaging Subset of Digital Database for Screening Mammography)
- InBreast (INbreast Breast Cancer Database)
- Private Database (Tampere dataset)

### Models

- Deep Learning: VGG16 variants, YaroslavNet
- Machine Learning: Gradient Boosting Classifier, Random Forest Classifier

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended for training)
- Conda or virtualenv for environment management

### Dependencies

Install the required packages using pip:

```bash
pip install tensorflow==2.8.0
pip install keras==2.8.0
pip install scikit-learn
pip install opencv-python
pip install pandas
pip install numpy
pip install matplotlib
pip install pydicom
pip install tqdm
pip install scikit-image
```

For GPU support, ensure CUDA and cuDNN are properly installed.

### Environment Setup

1. Create a conda environment:
```bash
conda create -n breast-cancer python=3.8
conda activate breast-cancer
```

2. Install dependencies as listed above.

## Usage

### Transfer Learning Module

Located in `TRANSFER_LEARNING/` directory.

#### Training

Run the training script with GPU diagnostics and model training:

```bash
cd TRANSFER_LEARNING/1st_config/CODES/
python training.py --gpus 0 --threads 4
```

Key parameters:
- `--gpus`: GPU device IDs (comma-separated)
- `--threads`: CPU threads for TensorFlow
- `--diagnostics-only`: Run GPU diagnostics without training

The script will:
1. Configure GPU environment
2. Run diagnostics (nvidia-smi, TensorFlow device check)
3. Train the model using transfer learning from VGG16
4. Save the best model and training metrics

Expected data structure:
```
Images/
├── TRAIN/
│   ├── neg/  # Negative/benign cases
│   └── pos/  # Positive/malignant cases
├── VALIDATION/
│   ├── neg/
│   └── pos/
└── TEST/
    ├── neg/
    └── pos/
```

### Inference Module

Located in `INFERENCE/` directory.

#### Running Inference

1. Prepare your test data in the expected format (see Data Structure section)

2. Run the inference script:

```bash
cd INFERENCE/CODES/
python inference.py
```

This will:
- Load trained models
- Generate predictions on test datasets
- Compute evaluation metrics (Accuracy, AUC, Sensitivity, Specificity)
- Save results to CSV files and generate comparison plots

#### Single Model Prediction

For individual model evaluation:

```bash
python generate_predictions.py \
    --exam-list-path /path/to/Metadata.pkl \
    --input-data-folder /path/to/Images \
    --prediction-file predictions.csv \
    --model model.h5 \
    --rescale-factor 0.003891 \
    --mean-pixel-intensity 44.4
```

### Radiomic Analysis Module

Located in `RADIOMIC_ANALISIS/` directory.

#### Feature Extraction and Training

1. Ensure feature CSV files are present:
   - `features_train.csv`
   - `features_val.csv`
   - `features_test.csv`
   - `labels_train.csv`, etc.

2. Run grid search for hyperparameter optimization:

```bash
cd RADIOMIC_ANALISIS/CODES/
python grid_search.py
```

3. Train and evaluate models:

```bash
python train_metrics.py
```

This module uses:
- PCA for dimensionality reduction
- StandardScaler for feature normalization
- Bootstrap AUC calculation with 95% confidence intervals
- Optimal threshold selection using G-Mean

## Data Structure

### For Transfer Learning

```
project_root/
├── Images/
│   ├── TRAIN/
│   │   ├── neg/  # PNG/JPG images
│   │   └── pos/
│   ├── VALIDATION/
│   │   ├── neg/
│   │   └── pos/
│   └── TEST/
│       ├── neg/
│       └── pos/
```

### For Inference

```
project_root/
├── Metadata.pkl          # Pickle file with exam metadata
├── Images/               # Directory with processed images
│   └── [image_id].png
```

Metadata.pkl structure:
- List of dictionaries containing exam information
- Each exam has L-CC, L-MLO, R-CC, R-MLO views
- Labels for malignancy (left_malignant, right_malignant)

### For Radiomics

CSV files with extracted features:
- `features_train.csv`: Training features
- `features_val.csv`: Validation features
- `features_test.csv`: Test features
- `labels_train.csv`: Training labels
- `labels_val.csv`: Validation labels
- `labels_test.csv`: Test labels

## Results

### Deep Learning Models Performance

| Model | Dataset | Accuracy | AUC | Sensitivity | Specificity |
|-------|---------|----------|-----|-------------|-------------|
| ddsm_vgg16_s10_512x1.h5 | CBIS-DDSM | 0.7973 | 0.8815 | 0.7911 | 0.8023 |
| inbreast_vgg16_512x1.h5 | CBIS-DDSM | 0.6952 | 0.7898 | 0.7490 | 0.6513 |
| ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5 | CBIS-DDSM | 0.8387 | 0.9002 | 0.7315 | 0.9261 |
| ddsm_YaroslavNet_s10.h5 | CBIS-DDSM | 0.7602 | 0.8269 | 0.6799 | 0.8257 |

### Radiomics Models Performance

| Model | AUC Train [IC95] | AUC Val [IC95] | AUC Test [IC95] | Acc Test | Sens Test | Spec Test |
|-------|------------------|----------------|-----------------|----------|-----------|-----------|
| GradientBoosting | 0.983 [0.979–0.987] | 0.894 [0.879–0.909] | 0.892 [0.877–0.907] | 0.832 | 0.833 | 0.831 |
| RandomForest | 1.000 [1.000–1.000] | 0.903 [0.889–0.917] | 0.901 [0.887–0.915] | 0.841 | 0.833 | 0.848 |

*Note: Results may vary based on data preprocessing and hyperparameter tuning.*

## Configuration

### Model Configurations

- **1st_config**: Basic transfer learning setup
- **2nd_config**: Alternative configuration (if available)

### Hyperparameters

Key hyperparameters are defined in the training scripts:
- Image size: 1152x896
- Batch size: 16
- Learning rates: 0.0002 (initial), 0.001 (all layers)
- Optimizer: Adam
- Weight decay: 0.0001

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure proper CUDA installation and compatible TensorFlow version
2. **Memory Errors**: Reduce batch size or use CPU if GPU memory is insufficient
3. **Import Errors**: Install missing dependencies or check Python path
4. **Data Format Issues**: Verify image formats (PNG/JPG) and directory structure

### Performance Tips

- Use multiple GPUs for faster training
- Preprocess images to reduce I/O overhead
- Monitor GPU utilization with nvidia-smi
- Use appropriate batch sizes based on available memory

## Authors

- [Your Name] - Thesis work on breast cancer detection
- Supervisor: [Supervisor Name]

## License

This project is part of an academic thesis. Please contact the authors for usage permissions.

## Acknowledgments

- CBIS-DDSM dataset providers
- InBreast database creators
- Open-source libraries: TensorFlow, scikit-learn, OpenCV