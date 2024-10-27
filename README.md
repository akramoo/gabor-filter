Here’s an updated version of the README file that includes the folder structure and explanations:

---

# Medical Image Dataset Processing for Pneumonia Detection

## Overview

This project focuses on preparing a chest X-ray image dataset for pneumonia detection. The process includes relabeling images, applying Gabor filters for texture extraction, and saving the features in a CSV file. This feature set enables efficient integration into machine learning pipelines for classification tasks. The method leverages Gabor filters due to their effectiveness in capturing texture patterns, which are crucial in differentiating healthy from pneumonia-affected lungs.

### Use Case

The code supports preprocessing a chest X-ray dataset for binary classification:
- **Classes**: "NORMAL" and "PNEUMONIA"

---

## Project Structure

```
.
├── Datasets                    # Folder containing the dataset of chest X-ray images
│   ├── NORMAL                  # Subfolder containing images labeled as NORMAL
│   └── PNEUMONIA               # Subfolder containing images labeled as PNEUMONIA
├── .gitignore                  # Git ignore file to exclude unnecessary files from version control
├── Datasets_features.csv       # CSV file containing extracted features from images
├── LICENSE                     # License for the project
├── README.md                   # Project documentation
├── dataset_relabeler.ipynb     # Jupyter notebook for relabeling dataset images
├── gabor_feature_extractor_to_csv.ipynb  # Jupyter notebook for applying Gabor filters and saving features to CSV
└── project_report.pdf          # Detailed report on the dataset processing and feature extraction
```

### Folder and File Explanations

- **`Datasets/`**: Main folder containing the image dataset, organized into two subdirectories:
  - **`NORMAL/`**: Contains chest X-ray images classified as "NORMAL."
  - **`PNEUMONIA/`**: Contains chest X-ray images classified as "PNEUMONIA."

- **`.gitignore`**: Specifies files and folders that should be ignored by Git, helping to keep the repository clean.

- **`Datasets_features.csv`**: A CSV file where the processed features from each image are stored. Each row contains texture features extracted using Gabor filters along with the label and unique identifier for each image.

- **`LICENSE`**: The license file outlines the terms of use, redistribution, and any restrictions associated with this project.

- **`README.md`**: This file provides a comprehensive overview of the project, its structure, and the code functionality.

- **`dataset_relabeler.ipynb`**: A Jupyter notebook containing code to relabel the images in the dataset. It renames each image file with a unique identifier that includes the class label and a counter.

- **`gabor_feature_extractor_to_csv.ipynb`**: A Jupyter notebook for applying Gabor filters to each image in the dataset to extract texture features. The processed features and labels are then saved to `Datasets_features.csv`.

- **`project_report.pdf`**: A detailed report on the dataset processing methodology, the application of Gabor filters, and the extracted feature dataset. This includes references to relevant resources.

---

## Code Explanation

### 1. Relabeling the Dataset

Each image is relabeled based on its class and a unique counter to ensure consistent and recognizable filenames.

```python
# Renaming images with unique identifiers
for sub_dir in sub_dirs:
    folder_path = os.path.join(base_dir, sub_dir)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                new_name = f"{sub_dir}_{global_count}.jpg"
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                global_count += 1
```

### 2. Defining and Applying Gabor Filters

Gabor filters are applied at multiple orientations, scales, and frequencies to capture texture patterns effectively.

- **Theta (Orientation)**: 0°, 45°, 90°, and 135°
- **Sigma (Scale)**: 1 and 3
- **Frequency**: 0.05 and 0.25

This configuration ensures the filters can detect texture variations relevant to lung condition analysis.

```python
# Creating Gabor filters with various parameters
kernels = []
for theta in range(4):
    theta = theta / 4.0 * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
```

### 3. Feature Extraction and Saving Data to CSV

Each image is processed to extract:
- **Mean**: Average intensity of the filtered image.
- **Variance**: Measures texture variation strength.

These features are saved in a CSV file, providing a structured dataset for machine learning models.

```python
# Feature extraction and saving to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["numero d'image", "caractéristique", "label"])

    for sub_dir in sub_dirs:
        folder_path = os.path.join(base_dir, sub_dir)
        
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    image = img_as_float(io.imread(image_path, as_gray=True))
                    feats = compute_feats(image, kernels)
                    numero_image = filename.split('_')[-1].split('.')[0]
                    label = filename.split('_')[0]
                    feats_str = str(feats.tolist())
                    writer.writerow([numero_image, feats_str, label])
```

---

## Summary

This code pipeline efficiently prepares a medical dataset for pneumonia detection by:
- **Relabeling images** for structured dataset organization.
- **Extracting Gabor texture features** that enhance the model's ability to distinguish between healthy and pneumonia-affected lungs.
- **Saving features and labels in CSV** format for easy integration with machine learning models.

This enriched, texture-focused dataset is ready for classification tasks, enabling a more accurate distinction between "NORMAL" and "PNEUMONIA" cases.

---