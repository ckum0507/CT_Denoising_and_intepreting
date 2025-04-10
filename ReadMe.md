<p align='center'>
Welcome to our Final Year project!!!
</p>
<p align='center'>
Comparision of all DnCNN iterations occuring Periodic Denoising
</p>

![Alt text](Outputs/Aneurysm_Brain/DnCNN_comparision.png)
<p align='center'>
Comparision of all Denoising stages
</p>

![Alt text](Outputs/Aneurysm_Brain/Overall_compoarision.png)

This project contains Denoising of CT images using a hybrid technique of Periodic Denoising and Poisson's Denoising using Denoising Convolutional Neural Network and HNIPM/PRBF algorithms and Interpret the abnormality using ResNet-50 and EfficientNet-B7 algorithms.

All the models are initially trained for Axial CT scan of Human Brain only
The Periodic DnCNN models are well trained with 3 different paramenters able to reduce noise at three different decreasing levels of noise.

The Interpreting Deep-Learning Models are trained with anotated dataset of Aneursym, Cancer, Tumour-Glioma, Tumour-Meningioma, Tumour-Pituitary and Normal Brain 

To run this program you can:
```  
  git clone https://github.com/ckum0507/CT_Denoising_and_intepreting.git
  cd CT_Denoising_and_intepreting
```
Introduce a virtual environment to install libraries:
```
  python  -m venv venv
```
Activating Virtual Environment:
```  
  venv/Scripts/activate # for Windows terminal
  source venv/bin/activate # for bash terminal
```
Installing python library files:
```
  pip install -r cuda-requirements.txt # to work in Dedicated NVidia GPU (requires NVidia GPU driver and CUDA app)
  pip install -r cpu-requirements.txt  # to work in CPU 
```
Running the file:
```
  python app.py
```
