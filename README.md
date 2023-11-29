# Deep-Spatio-Temporal
Code for [Deep Spatio-Temporal Wind Power Forecasting](https://arxiv.org/abs/2109.14530)
## How to use
The model is validated on two datasets.
### Wind power forecasting 
This dataset is from https://aml.engr.tamu.edu/book-dswe/dswe-datasets/. The data used here is Wind Spatio-Temporal Dataset2. Download data, put it into the './data' folder and rename it to 'wind_power.csv'. Then, run following
```
python train.py --name wind_power --epoch 300 --batch_size 20000 --lr 0.001 --k 5 --n_turbines 200
```

## References
* Jiangyuan Li, Mohammadreza Armandpour. (2022) "Deep Spatio-Temporal Wind Power Forecasting". IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
