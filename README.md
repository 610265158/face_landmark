# face_landmark

A simple face aligment method.


## introduction
This is the pytorch branch. The tf version is no longer maintained. 

It is simple and flexible, 
trained with **wingloss** , 
**multi task learning**, 

also with **data augmentation based on headpose and face attributes(eyes state and mouth state)**.

[CN blog](https://blog.csdn.net/qq_35606924/article/details/99711208)

And i suggest that you could try with another project,including face detect and keypoints, and some optimizations were made, u can check it there **[[pappa_pig_face_engine]](https://github.com/610265158/Peppa_Pig_Face_Engine).**

Contact me if u have problem about it. 2120140200@mail.nankai.edu.cn :)

demo pictures:

![samples](https://github.com/610265158/face_landmark/blob/master/figures/tmp_screenshot_18.08.20192.png)

![gifs](https://github.com/610265158/Peppa_Pig_Face_Engine/blob/master/figure/sample.gif)

this gif is from github.com/610265158/Peppa_Pig_Face_Engine )

pretrained model:

###### shufflenetv2_1.0

+ [baidu disk](https://pan.baidu.com/s/1MK3wI0nrZUOA8yU0ChWvBw)  (code 9x2m)



## requirment

+ pytorch

+ opencv

+ python 3.7


## useage

### train

1. I have made a new 68points dataset that collect from 300W and 300VW. Please download from [HERE]()
3. then, run:  `python train.py`

4. by default it trained with mobilenetv3

### visualization

```
python vis.py --model ./model/keypoints.pth
```







