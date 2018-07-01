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
* Volume normalize
* Time strech
* Audio file repeat, offset, and cut
* Pitch shift
* Noise

### MFCC
* sampling_rate
* audio_duration
* n_mfcc


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
### Data Preprocessing
#### Volume normalize
一開始，我們使用最一般的normalization
```python
def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5
```
但是後來想到這種方法有可能使得原本是0的地方(無聲)變成另一個constant，這對於mfcc有不好的影響，因為mfcc用了一種類似人耳處理音訊的方法，一個offset可能對於mfcc有不良的影響，所以我們改成下列的normalization
```python
def audio_norm(data):
    max_data = np.max(np.absolute(data))
    return data/(max_data+1e-6)*0.5
```
這種方法就單純把音量normalize，盡量讓音訊原汁原味地送給mfcc。得到的成果也是豐碩的。

| 方法 | testing error on mfcc |
| :-------: | :-----: |
| 一般的normalization | 0.857 |
| 只調音量的normalization | 0.882 |

直接進步了3%。另外我們也用這個方法應用在1D convolution的model，但是並沒有進步，我們的推測是其中的neuron自帶有offset的功能，所以能夠進而調整，不像2D 的model需要經過mfcc，故沒有顯著的進步效果。
#### Audio file repeat, offset, and cut
因為model在吃參數的時候，必須要是相同格式，我們必須把所有音檔都轉換成相同長度，我們選擇4秒當作一個標準長度，一開始我們處理的方式如下:
* 如果比4秒還長，我們在這個音檔中隨機取一個4秒的片段
* 如果比4秒還短，我們把音檔隨機放在一個4秒的某個位置，剩下補0

### Data Generator
#### Pitch shift
我們利用librosa內建的function `librosa.effects.pitch_shift` 去實現，一開始我們將所有音量都隨機調整正負一個8度，但是並沒有得到好的結果，後來我們覺得可能頻率移動太多，某些音檔可能已經變得無法辨識，所以我們從正負一個8度調整到正負0.3度，可是依然無法得到明顯進步的結果，所以我們就放棄這個方法了。
#### Noise
從課堂中我們學到，在data中適當的加入noise可以增加model的robusticity，所以我們加入最大音量5%的white noise，可惜的也沒有得到顯著的進步，之後我們把noise的level又調到1%，但是依然沒有得到進步的結果。我們推測所有音檔(包含testing、training)都收音蠻好的，幾乎沒甚麼雜訊，機器在判斷的時候，noise反而沒辦法得到好處。
## Conclusion

## Reference

- [Beginner's Guide to Audio Data](https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data)

  在這份kernel中的code裡面學到了很多音訊相關的處理方式.......

- [More To Come. Stay Tuned](https://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis)

- [Confusion Matrix on scikit-learn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)

  ​
