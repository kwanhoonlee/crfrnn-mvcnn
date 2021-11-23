# MVCNN with CRF-RNN
This is our implementation for the paper about MVCNN with CRF-RNN model for BIM:

Koeun Lee, Yongsu Yu, Daemok Ha, Bonsang Koo, Kwanhoon Lee (2021). [MVCNN with CRF-RNN for BIM](http://koreascience.or.kr/article/JAKO202130548390142.page) 


*Used the source code [MVCNN-Keras](https://github.com/Mannix1994/MVCNN-Keras) and [CRFRNN-Keras](https://github.com/sadeepj/crfasrnn_keras).

# Abstract
In order to maximize the use of BIM, all data related to individual elements in the model must be correctly assigned, and it is essential to check whether it corresponds to the IFC entity classification. However, as the BIM modeling process is performed by a large number of participants, it is difficult to achieve complete integrity. To solve this problem, studies on semantic integrity verification are being conducted to examine whether elements are correctly classified or IFC mapped in the BIM model by applying an artificial intelligence algorithm to the 2D image of each element. Existing studies had a limitation in that they could not correctly classify some elements even though the geometrical differences in the images were clear. This was found to be due to the fact that the geometrical characteristics were not properly reflected in the learning process because the range of the region to be learned in the image was not clearly defined. In this study, the CRF-RNN-based semantic segmentation was applied to increase the clarity of element region within each image, and then applied to the MVCNN algorithm to improve the classification performance. As a result of applying semantic segmentation in the MVCNN learning process to 889 data composed of a total of 8 BIM element types, the classification accuracy was found to be 0.92, which is improved by 0.06 compared to the conventional MVCNN.

# Requirements
* CUDA 10.0 (if you have NVIDIA GPU) 
* python 2.7 or python 3.5+ 
* tensorflow-gpu 1.12.0 
* nvidia-ml-py(for python 2.7)
* nvidia-ml-py3(for python 3.5+)
* make for CRF-RNN </br>

# Building CRF-RNN
```bash
cd MVCNN_with_CRFRNN/src/cpp
make
```
# Train
```bash
# for training dataset using only MVCNN
python3 train.py
# for training dataset using only MVCNN with CRF-RNN
python3 train_with_crf.py
```

# Evaluate
```bash
python3 evaluate.py
# or
python3 evaluate_with_crf.py
```

# Predict 
```bash
python3 predict.py

```

