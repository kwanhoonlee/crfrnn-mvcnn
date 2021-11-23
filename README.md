# crfrnn-mvcnn

This is the MVCNN with CRF-RNN model coding with keras.  
Used the source code [MVCNN-Keras](https://github.com/Mannix1994/MVCNN-Keras) and [CRFRNN-Keras](https://github.com/sadeepj/crfasrnn_keras).

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

