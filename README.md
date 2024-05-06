# Multimodal Pneumonia Detection Model

This collection of 3 Jupyter notebooks demonstrates the development of a deep learning model that integrates both image and textual data to diagnose pneumonia along with the two Uni-modal models that it is composed of. It leverages PyTorch, PyTorch Lightning, and the Hugging Face Transformers library to train a model using both chest X-ray images and associated clinical text.

## Abstract:
The efficient integration of multi-modal data is a cornerstone of robust deep learning appli-
cations, particularly in healthcare, where diverse patient information can significantly im-
pact diagnostic accuracy. The primary objective of this project is to develop a multi-modal
model that can integrate MIMIC IV CXR imaging data with corresponding radiology notes
to distinguish between ’pneumonia’ and ’not pneumonia’ cases. These non-pneumonia cases
are the absence of any abnormal pulmonary diseases classified by the ChexPert model. A
critical aspect of this project is confronting the significant class imbalance present in medical
datasets, which often feature a disproportionate number of non-disease instances(Welvaars
et al., 2023). Through extensive experiments on the medical dataset, we explore the be-
haviors and characteristics of the multi-modal models and observe relative improvements
to the robustness and reliability for a more balanced learning process. We particularly
focused on reducing the effects of class imbalance through sub-sampling and implementing
different loss functions. Our results demonstrate that multi-modal models, which integrate
both text and image data, were able to perform competitively with their single-modality
counterparts across a variety of metrics in a context where class imbalance was less signif-
icant. These findings hold value for those considering the deployment of machine learning
models in clinical settings, where accurate interpretation of multi-modal data is ideal for
patient outcomes

## Prerequisites

Before running this notebook, ensure you have the following:

- Python 3.8 or above
- PyTorch 1.8 or later
- PyTorch Lightning
- Hugging Face Transformers
- Matplotlib
- Seaborn
- Pandas
- NumPy
- Scikit-learn

You can install the necessary libraries via pip:

```bash
pip install torch torchvision torchaudio pytorch-lightning transformers matplotlib seaborn pandas numpy scikit-learn
```

GPU training is recommended for the code contained in this repo.

## Dataset

The model expects a dataset containing labeled images and their corresponding text descriptions. The dataset should be structured as a CSV file with at least the following columns:

- `image_path`: Path to the image file.
- `is_pneumonia`: Binary label indicating the presence of pneumonia (1 for positive, 0 for negative).
- `free_text`: Clinical notes or text associated with each image.



## Usage

1. **Data Preparation**: Load your data and split it into training, validation, and test sets.
> - Update the file path variables in the notebook to point to your dataset location. The two provided csv datasets are "final_cxr_free_text75.csv" and "final_cxr_free_text.csv" for the subsampled and full datasets.

> - As well, during the creation of the custom dataset for training and evaluation, ensure that the datapath to the images are correctly updated.

> - The structure of the CXR images directory should be /Dir_Name/(* images in .jpg). There should not be any nesting of subdirectories.
2. **Model Training**: Use the provided `MultimodalModel` `ImageModel` or `TextModel ` class to train the model on your dataset depending on which .ipynb notebook you are running. The provided jupyter notebooks both include the training loops defined for the full and subsampled datasets. 
> - Depending on which loss function you decide to train with, include the intended one in the pytorch lightning trainer instantiation within the script and name everything including logs and experiment results appropriately prior to running. 
> - When running focal loss in training, reminder that gamma and alpha values are at a default of 1.5 and 0.85 respectively, make alterations if different values are required. 
3. **Evaluation**: Evaluate the model using the validation and test datasets to understand its performance.
- If running Multimodal.ipynb, there are also Bimodal sensitivity tests that can be run after the other evaluations.

4. **Logging and Monitoring**: The training process logs metrics which can be viewed using TensorBoard.
> - The model_logs prior to training should be named appropriately for proper viewing. 

## Features

- **Custom Dataset Class**: Handles the loading and preprocessing/normalization of image and text data.
- **Multimodal Learning**: Combines features extracted from both images (using a modified ResNet50) and text (using BioBERT) to make predictions.
- **Evaluation Metrics**: Includes accuracy, precision, recall, F1 score, and AUROC for performance assessment.
- **Model Checkpoints, Early Stopping and Logging**: Integrated with PyTorch Lightning's logging and model checkpointing features along with early stopping callbacks from pytorch-lightning.


