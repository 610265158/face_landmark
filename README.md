# face_landmark
A simple face aligment method


## introduction
A simple face aligment method based on tensorflow. 
It is simple and flexible, trained with wingloss , multi task learning, also with data augmentation based on headpose and face attributes(eyes state and mouth state).

[CN blog](https://blog.csdn.net/qq_35606924/article/details/99711208)

And i suggest that you could try with another project,including face detect and keypoints, and some optimizations were made, u can check it there **[[pappa_pig_face_engine]](github.com/610265158/Peppa_Pig_Face_Engine).**

Contact me if u have problem about it. 2120140200@mail.nankai.edu.cn :)

demo pictures:

![samples](https://github.com/610265158/face_landmark/blob/master/figures/tmp_screenshot_18.08.20192.png)

![gifs](https://github.com/610265158/Peppa_Pig_Face_Engine/blob/master/figure/sample.gif)

this gif is from github.com/610265158/Peppa_Pig_Face_Engine, but it is the same model : )

pretrained model:

+ [baidu disk](https://pan.baidu.com/s/1jPW9cq9V9sJDrcrtcqpmLQ)  (code wd5g)
+ [google drive](https://drive.google.com/drive/folders/1YHtaLkalAqURbkIYYJBLf6HJZzd6vzOG?usp=sharing)



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

3. download the imagenet pretrained resnet_v1_50 model from [resnet50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)
or [mobilenetv2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz)
release it in the root dir

4. but if u want to train from scratch set config.MODEL.pretrained_model=None,

5. if recover from a completly pretrained model  set config.MODEL.pretrained_model='yourmodel.ckpt',config.MODEL.continue_train=True

then, run:

`python train.py`






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


- [ ]  2.pruning     

   train with resnet, then pruning, it should be better.



