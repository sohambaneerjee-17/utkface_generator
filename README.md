# utkface_generator
A Deep Convolutional GAN trained from scratch on UTKFace using PyTorch to generate synthetic human faces.
Face Generator from Scratch

A Deep Convolutional Generative Adversarial Network (DCGAN) built from scratch using PyTorch to generate synthetic human faces based on the UTKFace dataset.

## 🚀 About the Project
This project explores unsupervised representation learning using GANs. The model was trained end-to-end on the complete **UTKFace dataset (23,705 images)** for 150 epochs in Google Colab. Despite its compact footprint—the trained generator weighs only **14 MB**—the model successfully maps high-dimensional random noise into diverse, coherent 64x64 human faces featuring varying ages, ethnicities, and structures.

## 📊 Progression & Results
* **Early Stages (100 images / quick test):** Produced blurry, ghost-like facial silhouettes.
* **Mid Training (5,000 images):** Developed basic structural alignment and rudimentary facial features.
* **Final Model (Full 23,705 dataset, 150 Epochs):** Achieved high diversity, realistic skin texturing, and distinct age/ethnic variations without mode collapse.

## 🛠️ Tech Stack
* Language: Python
* Frameworks: PyTorch, Torchvision, NumPy
* Visualization: Matplotlib
* Environment:Google Colab (Cloud GPU) / Local Jupyter Notebook (Apple Silicon MPS support)
