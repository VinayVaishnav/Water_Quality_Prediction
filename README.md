# Water Quality Prediction (Intel OneAPI Online AI Hackathon)

## Overview
This repository contains the code and documentation for the "Water Quality Prediction" project, developed as part of the Intel OneAPI Online AI Hackathon. The project aims to predict the sustainability of water samples based on various features using machine learning models.  

## Important Links
- [Dataset (Kaggle)](https://www.kaggle.com/datasets/naiborhujosua/predict-the-quality-of-freshwater)
- [Preprocessed Dataset](https://drive.google.com/drive/folders/18Dg9FfzS2IPDBFLKcG8oY5yKIdNglbhn?usp=drive_link)
- [Ideation Presentation](https://docs.google.com/presentation/d/1q_5NuAXWf4dQiaDAx_z488UDvq2BjSQeXdBUrGjZCkQ/edit#slide=id.p)
- [Project Report](https://github.com/VinayVaishnav/Water_Quality_Prediction/blob/main/report.pdf)
- [Intel DevMesh Project](https://devmesh.intel.com/projects/water-quality-prediction-intel-oneapi-online-hackathon)
- [Solution Prototype Video (YouTube)](https://youtu.be/_Yrenj-xhrw?feature=shared)

## Problem Statement
Freshwater is a crucial natural resource, and ensuring its quality is essential for various aspects of human life and ecosystems. The goal of this project is to predict the sustainability of water samples for consumption using a provided dataset.

## Project Directory Structure
### Root Directory

| Directory/File                 | Description                                                |
|---------------------------------|------------------------------------------------------------|
| `models`                        | Directory containing the saved models. |
| `model_making_and_testing`      | Directory with Jupyter notebooks and preprocessed data.    |
| `report_water_quality_prediction.pdf`| Detailed report on the project.                             |
| `water_quality_prediction.py`   | Main Python script for running the water quality prediction.|

### `models` Directory

| File                | Description                                     |
|---------------------|-------------------------------------------------|
| `model_1.zip`       | Saved model 1.                                  |
| `model_2.zip`       | Saved model 2.                                  |
| `model_3.zip`       | Saved model 3.                                  |

### `model_making_and_testing` Directory

| Directory/File                                 | Description                                               |
|-------------------------------------------------|-----------------------------------------------------------|
| `classification.ipynb`                        | Notebook for classification models.                       |
| `data_analysis_and_visualizations.ipynb`       | Notebook for data analysis and visualizations.            |
| `preprocessed_water.csv`                       | Preprocessed dataset.                                     |
| `preprocessing.ipynb`                         | Notebook for data preprocessing.                          |
| `saving_final_model.ipynb`                    | Notebook for saving the final model.                      |
| `water.csv`                                    | Original dataset.                                        |

## Dependencies
- `Python 3.x`
- `intel-extension-for-pytorch==2.0.100`
- `matplotlib==3.7.1`
- `numpy==1.23.5`
- `pandas==2.0.1`
- `pytorch-tabnet==4.1.0`
- `scikit-learn==1.2.2`
- `scikit-learn-intelex==2023.2.1`
- `scipy==1.10.1`
- `seaborn==0.12.2`
- `torch==2.0.0+cpu`
- `xgboost==1.7.6`

## Usage
1. Clone the repository: `git clone https://github.com/VinayVaishnav/Water_Quality_Prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python3 water_quality_prediction.py`

## Data Analysis, Visualization, and Preprocessing
### Dataset Overview
The provided dataset comprises 59,56,842 data points with 22 feature columns and 1 column indicating target labels (sustainability of the sample). Key features include pH level, various chemical contents, color, turbidity, odor, conductivity, total dissolved solids, source, water and air temperatures, and date-time information.

### Key Observations
- The dataset is imbalanced with 69.69% samples labeled as not sustainable (Target: 0) and 30.31% as sustainable (Target: 1).
- Categorical features like source, month, day, and time of the day exhibit uniformity in the number of data points across categories.

### Issues Encountered
- Approximately 20 lakh rows with missing values out of the 59 lakh data points.
- High dimensionality leading to increased computation and model training time.

### Preprocessing Steps
1. **Dealing with Missing Values:**
   - For insignificant missing values, drop the corresponding rows.
   - Fill missing values with the overall mean of the feature column based on concentration graphs.

2. **Handling High Dimensionality:**
   - Utilized Intel AI Analytics Toolkit to optimize computations.

3. **After Preprocessing:**
   - Reduced to a binary classification problem with around 51 lakh data points.

## Technological Aspect and Methodology

### Intel AI Analytics Toolkit
The Intel AI Analytics Toolkit was employed to optimize the project's workflow. Notable packages used include:
- Intel Extensions for Scikit-learn and XGBoost
- oneDNN (Intel oneAPI Deep Neural Network Library)
- Intel Extension for PyTorch

### Methodology
- **Model Selection:**
   - Started with simpler models like Logistic Regression and Decision Tree for efficient results.
   - Gradually moved to more complex models including XGBoost, Multilayer Perceptron (MLP), and TabNet.

- **Model Evaluation:**
   - Dataset split into 70:10:20 (train: validation: test).
   - Evaluation metrics: F1 Score for binary classification.

### Results
| Model                   | Accuracy Score (on test set) | F1 Score (on test set) |
|-------------------------|-------------------------------|-------------------------|
| Logistic Regression     | 79.02%                        | 0.5931                  |
| Decision Tree           | 83.03%                        | 0.7179                  |
| XGBoost                 | 86.37%                        | 0.7961                  |
| MLP                     | 83.94%                        | 0.7581                  |
| TabNet                  | 87.28%                        | 0.8155                  |

### TabNet
TabNet, proposed by Google Cloud in 2019, provides a high-performance and interpretable tabular data deep learning architecture.  
Key features:
- Sequential attention mechanism for feature selection.
- Efficient training and high interpretability.
- Utilized for its stability in handling noisy data.

### Performance Improvement
- Ensembled multiple TabNet models for enhanced predictive accuracy and generalization.
- Achieved an **accuracy score of 87.39%** and an **F1 Score of 0.81786**.


## Role of Intel AI Analytics Toolkit

### Intel Extension for Scikit-Learn and XGBoost

- `sklearnex` is an extension of scikit-learn that optimizes machine learning algorithms for faster execution on multi-core processors.
- Components are replaced with `sklearnex` counterparts to reduce computation time while maintaining or improving model performance.
- `XGBoost`, a gradient boosting library, efficiently uses CPU cores and can run on GPUs, offering speed improvements for gradient boosting tasks.

### Intel Extension for PyTorch

- `intel_extension_for_pytorch` enhances PyTorch's speed by leveraging Intel's hardware acceleration capabilities.

### Advantages of using the toolkit

- Tools collectively reduce computation time and enhance performance.
- Easy integration into existing Python code with just a few extra lines.
- Faster iterations, improved scalability, and the ability to tackle high-dimensional datasets.
- Streamlined machine learning workflow for efficiency and scalability.

## References
- To know more about the types of extensions, packages, and libraries in the toolkit:  
    https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html#gs.4lj2sh  
    https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html#gs.4lj0kq  
- For installation of the Intel packages:  
    https://intel.github.io/scikit-learn-intelex/  
    https://pypi.org/project/scikit-learn-intelex/  
    https://pytorch.org/tutorials/recipes/recipes/intel_extension_for_pytorch.html  
- For understanding the workings of Intel packages:  
    https://youtu.be/vMZNYP4e2xo?si=Arw_ILgs_-l_RUka  
- Regarding TabNets:  
    https://www.geeksforgeeks.org/tabnet/  
    https://paperswithcode.com/method/tabnet  

## Authors
- [Vinay Vaishnav](mailto:vaishnav.3@iitj.ac.in): Pre-final Year (B.Tech. Electrical Engineering)
- [Tanish Pagaria](mailto:pagaria.2@iitj.ac.in): Pre-final Year (B.Tech. Artificial Intelligence & Data Science)  
(IIT Jodhpur Undergraduates)
