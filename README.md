Overview

This project demonstrates the classification of handwritten digits from the MNIST dataset using a deep learning neural network built with TensorFlow and Keras. The model achieves high accuracy in recognizing digits, making it a great example for beginners in machine learning and deep learning.

Features

Dataset: Uses the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9).

Model Architecture: A neural network with:

Input layer (flattened 28x28 images)

Two hidden layers with ReLU activation

Output layer with sigmoid activation

Training: The model is trained for 10 epochs, achieving a training accuracy of ~99% and a test accuracy of ~97%.

Visualization: Includes code to display sample images and their corresponding labels.

Requirements
Python 3.x

TensorFlow

Keras

NumPy

Matplotlib

Seaborn

OpenCV (for additional visualization, optional)

Installation
Clone the repository:

bash
git clone https://github.com/yourusername/mnist-digit-classification.git
cd mnist-digit-classification
Install the required packages:

bash
pip install tensorflow numpy matplotlib seaborn opencv-python
Usage
Open the Jupyter Notebook mnist_project_ML.ipynb.

Run the cells sequentially to:

Load and preprocess the MNIST dataset.

Build and compile the neural network.

Train the model on the training data.

Evaluate the model's performance on the test data.

Visualize predictions and sample images.

Results
Training Accuracy: ~99.04%

Test Accuracy: ~97.18%

The model performs well on both the training and test datasets, demonstrating its capability to generalize to unseen data.

Files
mnist_project_ML.ipynb: Jupyter Notebook containing the complete code for data loading, model building, training, and evaluation.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is open-source and available under the MIT License.

