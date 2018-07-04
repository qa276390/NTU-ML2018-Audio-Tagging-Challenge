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
在這個task中我們主要使用兩種Model，分別是1D-CNN以及2D-CNN on MFCC。一開始有考慮使用RNN進行訓練，但最後仍選擇使用CNN，主要是考慮到因為音訊檔某個程度上具有時序性的關係，但這次的task並非語意辨識那樣有次序調換影響結果的關係，因此透過CNN的filter就可以將其時序相關性表現出來。
### 1D Convolution
我們將音訊檔取sample之後直接將Raw Data餵進1D-CNN Model，中間經過多層架構並有MaxPooling提升訓練速度。而這樣的架構相當簡單，但缺點就是訓練時間非常長，因為我們並沒有對音訊檔做預處理，因此Data size極大，需透過data_generator每次load進Model中訓練，而file I/O次數變大幅增加，也拖垮整體訓練速度。

<img src="img/1D-Conv.jpg" width="450">

### 2D Convolution on MFCC
在這裡則是較常見的做法，我們先將data透過MFCC做預處理在在將它餵進2D-CNN Model中，此時就接近一個我們熟悉的圖像辨識問題。而好處在於透過MFCC處理的資料不但Size較小且較接近人耳辨識聲音的方式，因此在這個模型上可以得到相當不錯的結果。

<img src="img/2D-Conv.jpg" width="450">

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
音訊處理並沒有像圖片一樣可以直接丟進一個CNN Network去處理，中間我們遇到很多問題，像是音檔的長度範圍過大，很容易造成照顧了短檔，忽略的長檔，或是照顧了長檔，但是忽略的短檔，還有在normalize中間其實也存在跟圖片處理不同的狀況。從作業我們也知道，data argumentation的成長幅度是驚人的，所以我們一開始也就把重心放在data argumentation，果真如我們所料，data argumentation所帶來的成果是大的，接下來我們會介紹我們遇到的問題與解決的方法。
#### Sampling Rate
人耳一般能聽到的聲音範圍極限大約是20kHz，我們選用的取樣頻率是44kHz從Nyquist theory得到最大能表現的頻率是22kHz，這樣使得能表現的頻率略高於人耳能聽見的頻率，讓人耳接受到的資訊完整呈現給機器處理，
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
因為model在吃參數的時候，必須要是相同格式，我們必須把所有音檔都轉換成相同長度，我們選擇2秒當作一個標準長度，一開始我們處理的方式如下:

* 如果比2秒還長，我們在這個音檔中隨機取一個2秒的片段
* 如果比2秒還短，我們把音檔隨機放在一個2秒的某個位置，剩下補0

但是我們想到，如果音檔本身很短，是不是有造成整個音檔大部分都是0，所以我們改用以下方法

* 如果比2秒還長，我們在這個音檔中隨機取一個2秒的片段
* 如果比2秒還短，我們把音檔自己重複使得長度剛好超過2秒，之後再隨機取2秒的片段

這種方法可以使得音檔完整呈現在data裡面。

#### Audio duration
經過音檔重複的處理，我們顧及到了音檔過短的問題，但是還是有少部分音檔是遠遠大於2秒的，為了也讓長音檔也能更完整呈現原本的資訊，我們把音檔的標準長度從2秒提升到4秒，綜合上述對音檔資訊的表現得處理，我們得到以下結果：

| 方法 | testing error on mfcc |
| :-------: | :-----: |
| 原始方法 | 0.914 |
| 處理過後 | 0.925 |

得到顯著的進步！

### Data Generator
#### Pitch shift
我們利用librosa內建的function `librosa.effects.pitch_shift` 去實現，一開始我們將所有音量都隨機調整正負一個8度，但是並沒有得到好的結果，後來我們覺得可能頻率移動太多，某些音檔可能已經變得無法辨識，所以我們從正負一個8度調整到正負0.3度，可是依然無法得到明顯進步的結果，所以我們就放棄這個方法了。
#### Noise
從課堂中我們學到，在data中適當的加入noise可以增加model的robusticity，所以我們加入最大音量5%的white noise，可惜的也沒有得到顯著的進步，之後我們把noise的level又調到1%，但是依然沒有得到進步的結果。我們推測所有音檔(包含testing、training)都收音蠻好的，幾乎沒甚麼雜訊，機器在判斷的時候，noise反而沒辦法得到好處。
#### Time strech
我們另外想到的方法是，隨機調整音檔的長度，使其產生更多相似的data，我們使用的是`librosa`這個library內的`librosa.effects.time_stretch`這個function，這個function可以把音檔調快或調慢，我們一開始讓聲音快慢範圍在0.5倍至2倍之間隨機選擇，可是並沒有得到進步的結果，我們想到有可能是範圍過大，導致有些音檔的資訊完全扭曲，所以我們進一步調整音檔快轉/慢轉的係數，最後決定在1.1倍至0.9倍這個數字最適合。
一開始我們是每個音檔，產生2個隨機放慢或放快的的音檔，因為得到好的成果，我們進一步從隨機2個，改至隨機4個，實踐data argumentation的精神，果真也有顯著的進步。

| 方法 | testing error on mfcc(ensembled) |
| :-------: | :-----: |
| 使用原始data | 0.902 |
| 使用Time strech 隨機量=2 | 0.911 |
| 使用Time strech 隨機量=5 | 0.914 |

因為再大記憶題無法負荷了，所以我們並沒有再繼續增加data argumentation的量，不過我們相信，還是有進步空間的。

## Conclusion

## Reference

- [Beginner's Guide to Audio Data](https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data)

  在這份kernel中的code裡面學到了很多音訊相關的處理方式.......

- [More To Come. Stay Tuned](https://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis)

- [Confusion Matrix on scikit-learn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)

  ​
