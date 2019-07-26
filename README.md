# face_landmark
A simple face aligment method




# simple pose estimation


## introduction
A simple face aligment method based on tensorflow. 
it is simple and flexible.


the evaluation results are based on resnet with batchsize(1x64),pretrained model can be download from
https://pan.baidu.com/s/1cUqnf9BwUVkCy0iT6EczKA ( password ty4d )



## requirment

+ tensorflow1.12

+ tensorpack (for data provider)

+ opencv

+ python 3.6



## useage


### train

1. download all the 300W data set including the 300VW(unzip as images)

3. but if u want to train from scratch set config.MODEL.pretrained_model=None,

4. if recover from a completly pretrained model  set config.MODEL.pretrained_model='yourmodel.ckpt',config.MODEL.continue_train=True

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


### evaluation

when u get a trained model, run 
```
python tools/auto_freeze.py
```
it will produce detector.pb in ./model

then run
```
python model_eval/eval_by_gt.py

```



```

### visualization

```
python vis.py

```

TODO A face detector is needed.


### pruning

after get a trained model, run 
``` python pruning/filter_model.py``` 
then produce a pruning.npz file, change config.MODEL.pretrained_model='pruning.npz'

and modify the net structure in net,

