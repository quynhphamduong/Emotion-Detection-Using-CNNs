# Emotion Detection Using CNNs

This project implements a facial emotion recognition system using Convolutional Neural Networks (CNNs). It is trained on the FER2013 dataset and uses a modified VGG-16 architecture to classify seven different emotions from facial expressions.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Facial emotion recognition is a significant field in artificial intelligence, with applications in security, healthcare, education, and human-computer interaction. The project uses a CNN-based approach to classify images into seven emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.

## Dataset
The project uses the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), which contains **35,887 grayscale images (48x48 pixels)**, each labeled with one of seven emotions. The dataset is split into training and testing sets.

![FER2013 Dataset Sample](https://github.com/quynhphamduong/Emotion-Detection-Using-CNNs/blob/0da7d7dada9614e13fa37e00fce62aa51f44f04b/fer2013datset.png)

## Model Architecture
The model is based on the **VGG-16** architecture with modifications:
- **Convolutional layers** for feature extraction
- **MaxPooling layers** for dimensionality reduction
- **Global Average Pooling** to reduce overfitting
- **Fully connected layers** for classification
- **Softmax activation** for multi-class classification

![VGG-16 Architecture](https://github.com/quynhphamduong/Emotion-Detection-Using-CNNs/blob/0da7d7dada9614e13fa37e00fce62aa51f44f04b/vgg16.jpg)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/quynhphamduong/Emotion-Detection-Using-CNNs.git
cd Emotion-Detection-Using-CNNs
```

### 2. Set up the environment
```bash
python3 -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```

## Usage

### Train the model
Run the Jupyter Notebook `Build_the_model.ipynb` on Google Colab to train the model.

### Test the model
Use `TestModel.ipynb` to evaluate the model’s accuracy.

## Results
The model achieved **~60% accuracy** on the test set due to dataset limitations and model overfitting. Below is a confusion matrix showing class-wise performance:

![Confusion Matrix](https://github.com/quynhphamduong/Emotion-Detection-Using-CNNs/blob/0da7d7dada9614e13fa37e00fce62aa51f44f04b/confusion%20matrix.png)

![Accuracy](https://github.com/quynhphamduong/Emotion-Detection-Using-CNNs/blob/a2958e7de9576071c0ec208bfbbf76d3829d95d9/accuracy.png)
![Loss](https://github.com/quynhphamduong/Emotion-Detection-Using-CNNs/blob/a2958e7de9576071c0ec208bfbbf76d3829d95d9/loss.png)
Some correctly classified images:

![Correct Predictions](https://github.com/quynhphamduong/Emotion-Detection-Using-CNNs/blob/0da7d7dada9614e13fa37e00fce62aa51f44f04b/goodtest.png)

Some misclassified images:

![Incorrect Predictions](https://github.com/quynhphamduong/Emotion-Detection-Using-CNNs/blob/0da7d7dada9614e13fa37e00fce62aa51f44f04b/wrongtest.png)

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.



---



