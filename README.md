# face_landmark
A simple face aligment method


## introduction
A simple face aligment method based on tensorflow. 
it is simple and flexible.
![samples1](https://github.com/610265158/face_landmark/blob/master/figures/tmp_screenshot_18.08.2019.png)
![samples2](https://github.com/610265158/face_landmark/blob/master/figures/tmp_screenshot_18.08.20192.png)

pretrained model can be download from
[baidudisk](https://pan.baidu.com/s/1NRneEVvfYRiTmgOD8_T-KA) (code hg7w)


## requirment

+ tensorflow1.12    (tensorflow1.14 if mix_precision turns on)

+ tensorpack (for data provider)

+ opencv

+ python 3.6


## useage

### train

1. download all the 300W data set including the 300VW(parse as images, and make the label the same formate as 300W)
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
release it in the root dir

4. but if u want to train from scratch set config.MODEL.pretrained_model=None,

5. if recover from a completly pretrained model  set config.MODEL.pretrained_model='yourmodel.ckpt',config.MODEL.continue_train=True

then, run:

`python train.py`



#### ** CAUTION， WHEN USE TENSORPACK FOR DATA PROVIDER, some change is needed. **
#### in lib/python3.6/site-packages/tensorpack/dataflow/raw.py ,line 71-96. to make the iterator unstoppable, change it as below. so that we can keep trainning when the iter was over. contact me if u have problem about the codes : )
```
 71 class DataFromList(RNGDataFlow):
 72     """ Wrap a list of datapoints to a DataFlow"""
 73 
 74     def __init__(self, lst, shuffle=True):
 75         """
 76         Args:
 77             lst (list): input list. Each element is a datapoint.
 78             shuffle (bool): shuffle data.
 79         """
 80         super(DataFromList, self).__init__()
 81         self.lst = lst
 82         self.shuffle = shuffle
 83     
 84     #def __len__(self):
 85     #    return len(self.lst)
 86 
 87     def __iter__(self):
 88         if not self.shuffle:
 89             for k in self.lst:
 90                 yield k
 91         else:
 92             while True:
 93                 idxs = np.arange(len(self.lst))
 94                 self.rng.shuffle(idxs)
 95                 for k in idxs:
 96                     yield self.lst[k]
```



### convert model
run `python tools/auto_freeze.py` produce keypoint.pb


### visualization

```
python vis.py

```

TODO: 
A face detector is needed.
pruning



