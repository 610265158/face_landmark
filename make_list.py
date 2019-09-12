import os
import random



def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):

            # if s == "pts":
            #     continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList
data_dir='PUB'      ########points to your director


pic_list=[]
GetFileList(data_dir,pic_list)

pic_list=[x for x in pic_list if '.jpg' in x or 'png' in x or 'jpeg' in x  ]

random.shuffle(pic_list)
ratio=0.9
train_list=pic_list[:int(ratio*len(pic_list))]
val_list=pic_list[int(ratio*len(pic_list)):]

# train_list=[x for x in pic_list if '300W/' not in x]
# val_list=[x for x in pic_list if '300W/' in x]

train_file=open('./train.txt',mode='w')
val_file=open('./val.txt',mode='w')


for pic in train_list:

    tmp_str=pic+'|'

    pts=pic.rsplit('.',1)[0]+'.pts'
    if os.access(pic,os.F_OK) and  os.access(pts,os.F_OK):
        try:
            with open(pts) as p_f:
                labels=p_f.readlines()[3:-1]
            for _one_p in labels:
                xy = _one_p.rstrip().split(' ')
                tmp_str = tmp_str + xy[0] + ' ' + xy[1] + ' '
            tmp_str = tmp_str + '\n'
            train_file.write(tmp_str)
        except:
            print(pic)


for pic in val_list:
    tmp_str=pic+'|'

    pts=pic.rsplit('.',1)[0]+'.pts'
    if os.access(pic,os.F_OK) and  os.access(pts,os.F_OK):
        try:
            with open(pts) as p_f:
                labels=p_f.readlines()[3:-1]
            for _one_p in labels:
                xy = _one_p.rstrip().split(' ')
                tmp_str = tmp_str + xy[0] + ' ' + xy[1] + ' '
            tmp_str = tmp_str + '\n'
            val_file.write(tmp_str)
        except:
            print(pic)
