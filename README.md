# Planet: Understanding the Amazon from Space

Every minute, the world loses an area of forest the size of 48 football fields. And deforestation in the Amazon Basin accounts for the largest share, contributing to reduced biodiversity, habitat loss, climate change, and other devastating effects. But better data about the location of deforestation and human encroachment on forests can help governments and local stakeholders respond more quickly and effectively.

Planet, designer and builder of the world’s largest constellation of Earth-imaging satellites, will soon be collecting daily imagery of the entire land surface of the earth at 3-5 meter resolution. While considerable research has been devoted to tracking changes in forests, it typically depends on coarse-resolution imagery from Landsat (30 meter pixels) or MODIS (250 meter pixels). This limits its effectiveness in areas where small-scale deforestation or forest degradation dominate.

Furthermore, these existing methods generally cannot differentiate between human causes of forest loss and natural causes. Higher resolution imagery has already been shown to be exceptionally good at this, but robust methods have not yet been developed for Planet imagery. 

## The data

### The label file

The label file consists of 40479 rows, each containing the labels of each image separated by white space

```0    train_0                               haze primary
1    train_1            agriculture clear primary water
2    train_2                              clear primary
3    train_3                              clear primary
4    train_4  agriculture clear habitation primary road
```

### What the data looks like

The first image of each row is the actual dataset image, associated with the given ID in the title, the two subsequent images of each row are examples of the data agumentation used during training and TTA (Test Time Augmentation)

![samplesperclass](http://i.imgur.com/y7bCKgr.jpg)
![samplesperclass](http://i.imgur.com/ksSC6ja.jpg)
![samplesperclass](http://i.imgur.com/L1rRJrW.jpg)
![samplesperclass](http://i.imgur.com/zvbvIM5.jpg)
![samplesperclass](http://i.imgur.com/zSkydpk.jpg)
![samplesperclass](http://i.imgur.com/Un7VqAT.jpg)
![samplesperclass](http://i.imgur.com/8Niu45O.jpg)
![samplesperclass](http://i.imgur.com/uVx4XJW.jpg)
![samplesperclass](http://i.imgur.com/783oE6n.jpg)
![samplesperclass](http://i.imgur.com/J249CM2.jpg)
![samplesperclass](http://i.imgur.com/qfzOEmW.jpg)
![samplesperclass](http://i.imgur.com/oPfTSaO.jpg)
![samplesperclass](http://i.imgur.com/VpvCY68.jpg)
![samplesperclass](http://i.imgur.com/Y8DA1MK.jpg)
![samplesperclass](http://i.imgur.com/5QSMb1a.jpg)
![samplesperclass](http://i.imgur.com/atlZHdL.jpg)

### Occurrences per category

![samplesperclass](http://i.imgur.com/GiuDYx8.png)

## Evaluation

Submissions will be evaluated based on their mean F2 score. The F score, commonly used in information retrieval, measures accuracy using the precision P and recall R. Precision is the ratio of true positives (TP) to all predicted positives (TP + FP). Recall is the ratio of true positives to all actual positives (TP + FN). The F2 score is given by

<p align="center"> 
<img src="http://i.imgur.com/7DRo4Vw.jpg">
</p>

Note that the F2 score weights recall higher than precision. The mean F2 score is formed by averaging the individual F2 scores for each row in the test set.

<p align="center"> 
<img src="http://i.imgur.com/q0hfvRt.png">
</p>

## Results

<p align="center"> 
<img src="http://i.imgur.com/epcSoEU.jpg">
</p>

The final result is able to reach ~0.927 F2 score on the private leaderboard at the ![competition's page](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)
