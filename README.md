# face_landmark
A simple face aligment method


## introduction
A simple face aligment method based on tensorflow. 
It is simple and flexible, trained with wingloss , multi task learning, also with data augmentation based on headpose and face attributes(eyes state and mouth state).

[CN blog](https://blog.csdn.net/qq_35606924/article/details/99711208)

And i suggest that you could try with another project,an algorithm including face detect and keypoints, and some optimizations were made. Check it there **[[pappa_pig_face_engine]](github.com/610265158/Peppa_Pig_Face_Engine).**

Contact me if u have problem about it. 2120140200@mail.nankai.edu.cn :)

demo pictures:

![samples](https://github.com/610265158/face_landmark/blob/master/figures/tmp_screenshot_18.08.20192.png)

![gifs](https://github.com/610265158/Peppa_Pig_Face_Engine/blob/master/figure/sample.gif)

this gif is from github.com/610265158/Peppa_Pig_Face_Engine, but it is the same model : )

pretrained model:

+ [baidu disk](https://pan.baidu.com/s/1jPW9cq9V9sJDrcrtcqpmLQ)  (code wd5g)
+ [google drive](https://drive.google.com/open?id=1YHtaLkalAqURbkIYYJBLf6HJZzd6vzOG)



## requirment

+ tensorflow1.14    (tensorflow 1.14 at least if mix_precision turns on)

+ tensorpack (for data provider)

+ opencv

+ python 3.6


## useage

### train

1. download all the [300W](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) data set including the [300VW](https://ibug.doc.ic.ac.uk/resources/300-VW/)(parse as images, and make the label the same formate as 300W)
```
├── 300VW
│   ├── 001_annot
│   ├── 002_annot
│       ....
├── 300W
│   ├── 01_Indoor
│   └── 02_Outdoor
├── AFW
│   └── afw
├── HELEN
│   ├── testset
│   └── trainset
├── IBUG
│   └── ibug
├── LFPW
│   ├── testset
│   └── trainset
```

2. run ` python make_list.py` produce train.txt and val.txt
(if u like train u own data, u should prepare the data like this:
`****.jpg| x1 y1 x2 y2 x3 y3...` 


3. then, run:  `python train.py`

4. by default it trained with shufflenetv2_1.0, if u like want train with resnet,do as follow:

     4.1 download pretrained model [resnet50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)
    
     4.2 change train_config as config.MODEL.net_structure='resnet_v1_50',config.MODEL.pretrained_model='resnet_v1_50.ckpt',
        it uses the first three blocks, so it is still fast, a pruning may achieve a better one.







### convert model
After training, convert the model to pb file and visualization.

run `python tools/auto_freeze.py` produce keypoint.pb


### visualization

```
python vis.py

```

### TODO: 
- [x] A face detector is needed.

  [dsfd_tensorflow](https://github.com/610265158/DSFD-tensorflow)
  
  [faceboxes-tensorflow](https://github.com/610265158/faceboxes-tensorflow)
          
  [pappa_pig_face_engine](github.com/610265158/Peppa_Pig_Face_Engine)


- [x]  train with resnet     

- [ ]  then pruning resnet, it should be faster.

- [ ]  transfer to tensorflow 2.0

