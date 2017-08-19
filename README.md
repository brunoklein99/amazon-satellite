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

A few classes don't have a strong represenation in the dataset for the CNN to be able to robustly classify them, but since they are also under-represented in the test set, this shouldn't be a problem.

## Evaluation

Submissions will be evaluated based on their mean F2 score. The F score, commonly used in information retrieval, measures accuracy using the precision P and recall R. Precision is the ratio of true positives (TP) to all predicted positives (TP + FP). Recall is the ratio of true positives to all actual positives (TP + FN). The F2 score is given by

<p align="center"> 
<img src="http://i.imgur.com/7DRo4Vw.jpg">
</p>

Note that the F2 score weights recall higher than precision. The mean F2 score is formed by averaging the individual F2 scores for each row in the test set.

<p align="center"> 
<img src="http://i.imgur.com/q0hfvRt.png">
</p>

The chart above illustrates the fact that it is better for us to tune our decision thresholds to reach a higher Recall instead of a higher Precision.

## Results

<p align="center"> 
<img src="http://i.imgur.com/epcSoEU.jpg">
</p>

The final result is able to reach ~0.927 F2 score on the private leaderboard at the ![competition's page](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

The irregular accuracy and loss are caused by the dynamic data augmentation nature of the training process.

## Analysis

### Biggest errors per category

<p align="center"> 
<img src="http://i.imgur.com/IsfyBRC.jpg">
</p>
<p align="center"> 
<img src="http://i.imgur.com/OGRziiD.jpg">
</p>
<p align="center"> 
<img src="http://i.imgur.com/Uy5Axuq.jpg">
</p>
<p align="center"> 
<img src="http://i.imgur.com/fRkulT4.jpg">
</p>
<p align="center"> 
<img src="http://i.imgur.com/P5M0k0o.jpg">
</p>
<p align="center"> 
<img src="http://i.imgur.com/NCclqFV.jpg">
</p>
<p align="center"> 
<img src="http://i.imgur.com/UgB2HZl.jpg">
</p>
<p align="center"> 
<img src="http://i.imgur.com/DJ33KlS.jpg">
</p>
<p align="center"> 
<img src="http://i.imgur.com/NhDeNuK.jpg">
</p>
 
It's interesting to see some images are clearly mislabeled, such as `agriculture`, `clear`, `habitation`, `road` others false negatives have just a glimpse of their feature, such as `partly_clouy` and `selective_logging` and some are honest mistakes such as `water`, `bare_ground` and `blooming`. The mislabeling is probably because the images are annotated in batches using a broader region from the one represented in the chips.

### Visualising the CNN's decision with "VisualBackProp"

Below we retrived the strongest true positive for each category and plotted a visualization that helps see how the CNN decided the outcome for that classification.

The implementation is almost entirely based on [VisualBackProp: efficient visualization of CNNs](https://arxiv.org/abs/1611.05418v3), but some ideas were gathered from [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901v3)

The brighest the spot, the more it contributed for the final result.

<p align="center"> 
<img src="http://i.imgur.com/w1ioVxP.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/qMkue0n.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/Erob9LV.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/fELVXBp.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/Dnwkx5z.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/QqHui4H.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/9clgdGY.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/HFgSfxY.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/zfkVojq.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/hGCS5xN.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/H5NIuGd.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/Uw0uIRy.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/l6IiMg0.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/XXiB8FD.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/rblmfxp.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/5jVkwZO.png">
</p>
<p align="center"> 
<img src="http://i.imgur.com/w5sw0em.png">
</p>
