# face_landmark
A simple face aligment method, based on pytorch


## introduction
This is the pytorch branch.
 
It is simple and flexible, trained with wingloss , multi task learning, also with data augmentation based on headpose and face attributes(eyes state and mouth state).

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

2. run ` python make_json.py` produce train.json and val.json
(if u like train u own data, please read the json produced , it is quite simple)

3. then, run:  `python train.py`

4. by default it trained with shufflenetv2_1.0


### visualization

```
python vis.py --model ./model/keypoints.pth
```








