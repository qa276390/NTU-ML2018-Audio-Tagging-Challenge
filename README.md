# Machine Learning (2018,Spring) Final Project

Team Member:

r06942018 何適楷   b04505021 黃廉弼  b04505026 蔡仲閔  b04901165 陳致維

## Dependency

This final project is dependent on Python 

- keras == 2.1.6
- librosa == 0.6.1
- numpy == 1.14.0
- pandas == 0.22.0
- matplotlib == 2.1.2
- requests == 2.18.4
- simplejson == 3.15.0

## Usage

### Data Preprocessing (Data Generator) 

To use the data generator to append the dataset, do:

```shell
# To use the data generator you should decide 
# 2 paraneters : strech, num
python3 ./final/src/data_gen.py --strech 1.1 --num 5
```

### Training

To train a model, do:

```shell
# To use the train you should give the model you want to use:
cd ./final/src && ./train.sh 1d_conv
cd ./final/src && ./train.sh 2d_mfcc
```


### Predict

```shell
cd ./final/src && ./predict.sh ./final/model/1d_conv ./final/model/2d_mfcc
```


### Crawler the Rank on Kaggle

To check the rank in NTU, we write a python crawler:

``` shell
python3 ./final/ranking.py
```

## Model

### 1D Convolution



### 2D Convolution on MFCC





## Reference

- [Beginner's Guide to Audio Data](https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data)

- [More To Come. Stay Tuned](https://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis)

- [Confusion Matrix on scikit-learn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)

  ​
