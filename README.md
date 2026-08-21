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
##   Acknowledgments
* **UTKFace Dataset:** Sourced and mirrored via Hugging Face community repositories. Original dataset creators: Zhifei Zhang, Yang Song, and Hairong Qi (*CVPR 2017*).





---------------------------------------------------

##  Features
* **Generates Brand New Faces:** Uses trained generator weights to dream up completely unique faces on the fly.
* **PyTorch Pipeline:** Clean implementation of a DCGAN architecture optimized for image generation.
* **Pre-trained Weights Included:** Ready-to-use generator model weights so you don't have to retrain from scratch.

---

## 📂 Project Structure
```text
utkface-dcgan-pytorch/
│
├── models/
│   └── utkface_generator_final.pth    # Pre-trained generator model weights
├── notebooks/
│   └── utkface_generator.ipynb        # Main training and inference notebook
├── LICENSE                            # MIT License
└── README.md                          # Project documentation
```
**🛠️ How to Run Locally**
If you want to download this repository and run the pre-trained model on your local machine to generate faces, follow these steps:
1. Clone the Repository
Bash
git clone [https://github.com/sohambanneerjee-17/utkface-dcgan-pytorch.git](https://github.com/sohambanneerjee-17/utkface-dcgan-pytorch.git)
cd utkface-dcgan-pytorch

**2. Install Dependencies**
Make sure you have Python and PyTorch installed, along with the required libraries:
Bash
pip install torch torchvision matplotlib numpy


3. Load and Test the Generator
You can load the saved model weights (models/utkface_generator_final.pth) in a Python script or Jupyter notebook to generate new faces:
Python
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# Load the trained weights onto CPU or CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your Generator architecture and apply state dict
# netG = Generator().to(device)
netG.load_state_dict(torch.load('models/utkface_generator_final.pth', map_location=device))
netG.eval()

# Generate random noise and create new faces
with torch.no_grad():
    noise = torch.randn(16, 100, 1, 1, device=device)
    fake_faces = netG(noise).detach().cpu()

# Plot the generated faces
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(fake_faces, padding=2, normalize=True), (1, 2, 0)))
plt.show()
