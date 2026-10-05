# Week 5: LSTM on Sequential Digits

This project classifies handwritten digits with a long short term memory network (LSTM) built in PyTorch. It is the Week 5 recurrent neural network assignment for TECH 405.

## Problem

The scikit-learn digits dataset has 1,797 images of the digits 0 to 9, with between 174 and 183 images per class. Each image is 8 by 8 pixels with integer values from 0 to 16. Every image is read row by row, so it becomes a sequence of 8 time steps with 8 features per step. The LSTM reads the rows in order, and a linear layer classifies its final hidden state.

## Model and setup

| Item | Value |
| --- | --- |
| Model | `nn.LSTM(8, 128)` followed by `nn.Linear(128, 10)` |
| Trainable parameters | 71,946 |
| Input scaling | Pixel values divided by 16 |
| Split | Stratified 80/20 train and test with seed 42, then 10% of the training part held out for validation (1,293 train, 144 validation, 360 test) |
| Loss and optimizer | Cross entropy, Adam, learning rate 0.003 |
| Batch size and epochs | 64 and 40, on a CPU |
| Evaluated model | The model from the final epoch, with the test set used once |

## Results

| Measure | Value |
| --- | --- |
| Baseline | 10% |
| Target | 93% |
| Test accuracy | 95.56% (344 of 360 correct) |

The target was met. The digits 2, 3, 4, and 7 were classified perfectly, and 7 of the 16 errors were 5s predicted as 8.

## Limitations

* Validation loss became unstable near the end of training (0.1430 at epoch 35, then 0.2804 at epoch 40), which points to mild overfitting on a small training set.
* The results come from one run with one seed. With 360 test images, an approximate 95% confidence interval for the accuracy is 93.4% to 97.7%.
* The split is random, so images from the same writer may appear in both the training and test sets.
* Only an LSTM was trained, so there is no comparison with a simple RNN.

## How to run

1. Install the packages:

```
python -m pip install -r requirements.txt
```

2. Open the notebook and choose Run all:

```
jupyter notebook week5_lstm_digits.ipynb
```

The dataset loads from scikit-learn, so nothing needs to be downloaded separately. The notebook also runs in Google Colab, where PyTorch is already installed. Training takes under a minute on a CPU.

## Files

* `week5_lstm_digits.ipynb`: the full notebook with outputs
* `Week5_RNN_Report.docx`: the written report
* `requirements.txt`: Python packages

## References

Alpaydin, E., & Kaynak, C. (1998). *Optical recognition of handwritten digits* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations*. https://doi.org/10.48550/arXiv.1412.6980

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., . . . Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*. https://doi.org/10.48550/arXiv.1912.01703

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*(85), 2825–2830. https://doi.org/10.48550/arXiv.1201.0490
