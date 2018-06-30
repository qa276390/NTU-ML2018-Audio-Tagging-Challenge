# Final Project Report

Team Member:

r06942018 何適楷  b04505021 黃廉弼 b04505026 蔡仲閔 b04901165 陳致維

## Introduction & Motivation 

由於智能管家的發展，聲音辨識算是現今ML的一大應用。

而音訊處理中，傅立葉轉換可說是其中相當重要的一個環節，即工學院基礎課程不斷出現的章節，現在相當是能應用所學的好機會。

加上本學期課程作業中沒有出現過以音訊處理為主題的作業，因此一個出於一個探索未知領域的好奇心，大家共識決定選擇音訊處理的題目當作報告主題。

為了探討各種樂器(廣泛來說)的不同，我們希望透過課程所學到的機器學習方法，來建立一定可靠度的模型能辨識出不同樂器的種類。並透過與kaggle平台上的高手們交流的過程得到良性的競爭。

## Data Preprocessing/Feature Engineering 

### Data Generator
Volume normalize
Time strech
Audio file repeat, offset, and cut
Pitch shift
Noise

### MFCC
sampling_rate
audio_duration
n_mfcc


## Model Description (At least two different models)

### 1D Convolution

### 2D Convolution on MFCC

```python
可以放一些程式碼
```



![]() 或Tensor Board的圖片

### Comparison

|                        | val_ACC | val_loss | ACC  | loss |
| ---------------------- | ------- | -------- | ---- | ---- |
| 1D Convolution         |         |          |      |      |
| 2D Convolution on MFCC |         |          |      |      |

## Experiment and Discussion

###Data Generator

### Normalization



## Conclusion

## Reference

- [Beginner's Guide to Audio Data](https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data)

  在這份kernel中的code裡面學到了很多音訊相關的處理方式.......

- [More To Come. Stay Tuned](https://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis)

- [Confusion Matrix on scikit-learn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)

  ​
