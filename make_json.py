import os
import random
import numpy as np
import json
import traceback

'''
i decide to merge more data from CelebA, the data anns will be complex, so json maybe a better way. 
'''





data_dir='/media/lz/73abf007-eec4-4097-9344-48d64dc62346/coco_data/facelandmark/PUB'      ########points to your director,300w
celeba_data_dir='/media/lz/73abf007-eec4-4097-9344-48d64dc62346/coco_data/CELEBA'                      ########points to your director,CELEBA


train_json='train.json'
val_json='val.json'



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




pic_list=[]
GetFileList(data_dir,pic_list)

pic_list=[x for x in pic_list if '.jpg' in x or 'png' in x or 'jpeg' in x  ]

random.shuffle(pic_list)
ratio=0.9
train_list=pic_list[:int(ratio*len(pic_list))]
val_list=pic_list[int(ratio*len(pic_list)):]

# train_list=[x for x in pic_list if '300W/' not in x]
# val_list=[x for x in pic_list if '300W/' in x]


train_json_list=[]
for pic in train_list:
    one_image_ann={}

    ### image_path
    one_image_ann['image_path']=pic

    #### keypoints
    pts=pic.rsplit('.',1)[0]+'.pts'
    if os.access(pic,os.F_OK) and  os.access(pts,os.F_OK):
        try:
            tmp=[]
            with open(pts) as p_f:
                labels=p_f.readlines()[3:-1]
            for _one_p in labels:
                xy = _one_p.rstrip().split(' ')
                tmp.append([float(xy[0]),float(xy[1])])

            one_image_ann['keypoints'] = tmp

            label = np.array(tmp).reshape((-1, 2))
            bbox = [float(np.min(label[:, 0])), float(np.min(label[:, 1])), float(np.max(label[:, 0])), float(np.max(label[:, 1]))]
            one_image_ann['bbox'] = bbox
            
            
            ### placeholder
            one_image_ann['attr'] = None
    
            train_json_list.append(one_image_ann)
        except:
            print(pic)
            traceback.print_exc()

   

with open(train_json,'w') as f:
    json.dump(train_json_list, f,indent=2)


val_json_list=[]
for pic in val_list:
    one_image_ann={}

    ### image_path
    one_image_ann['image_path']=pic

    #### keypoints
    pts=pic.rsplit('.',1)[0]+'.pts'
    if os.access(pic,os.F_OK) and  os.access(pts,os.F_OK):
        try:
            tmp=[]
            with open(pts) as p_f:
                labels=p_f.readlines()[3:-1]
            for _one_p in labels:
                xy = _one_p.rstrip().split(' ')
                tmp.append([float(xy[0]),float(xy[1])])

            one_image_ann['keypoints'] = tmp


            label = np.array(tmp).reshape((-1, 2))
            bbox = [float(np.min(label[:, 0])), float(np.min(label[:, 1])), float(np.max(label[:, 0])), float(np.max(label[:, 1]))]
            one_image_ann['bbox'] = bbox
            ### placeholder
            one_image_ann['attr'] = None

            val_json_list.append(one_image_ann)

        except:
            print(pic)


    

with open(val_json,'w') as f:
    json.dump(val_json_list, f,indent=2)









################# to process the CELEBA

ann_dir=os.path.join(celeba_data_dir,'Anno')
bbox_ann=ann_dir=os.path.join(ann_dir,'list_attr_celeba.txt')


with open(bbox_ann) as p_f:
    lines = p_f.readlines()[2:]




'''
5_o_Clock_Shadow 22516 180083
Arched_Eyebrows 54090 148509
Attractive 103833 98766
Bags_Under_Eyes 41446 161153
Bald 4547 198052
Bangs 30709 171890
Big_Lips 48785 153814
Big_Nose 47516 155083
Black_Hair 48472 154127
Blond_Hair 29983 172616
Blurry 10312 192287
Brown_Hair 41572 161027
Bushy_Eyebrows 28803 173796
Chubby 11663 190936
Double_Chin 9459 193140
Eyeglasses 13193 189406
Goatee 12716 189883
Gray_Hair 8499 194100
Heavy_Makeup 78390 124209
High_Cheekbones 92189 110410
Male 84437 118162
Mouth_Slightly_Open 97942 104657
Mustache 8417 194182
Narrow_Eyes 23329 179270
No_Beard 169158 33441
Oval_Face 57567 145032
Pale_Skin 8701 193898
Pointy_Nose 56210 146389
Receding_Hairline 16163 186436
Rosy_Cheeks 13315 189284
Sideburns 11449 191150
Smiling 97669 104930
Straight_Hair 42222 160377
Wavy_Hair 64744 137855
Wearing_Earrings 38276 164323
Wearing_Hat 9818 192781
Wearing_Lipstick 95715 106884
Wearing_Necklace 24913 177686
Wearing_Necktie 14732 187867
Young 156734 45865
'''



with open(train_json,'r') as f:
    train_json_list=json.load(f)
for line in lines:
    one_image_ann={}
    ann=line.rstrip().split(' ')
    ann=[x for x in ann if  x != '']
    one_image_ann['image_path']=os.path.join(celeba_data_dir,'Img','img_align_celeba',ann[0])


    one_image_ann['bbox']=[40,80,135,185]
    one_image_ann['keypoints']=None

    ##we select some attr, eyeglasses, gender and smile

    eyeglasses = 0 if int(ann[16]) < 0 else 1
    gender=0 if int(ann[21])<0 else 1
    Mouth_Slightly_Open=0 if int(ann[22])<0 else 1
    smile = 0 if int(ann[32]) < 0 else 1


    one_image_ann['attr']=[eyeglasses,gender,Mouth_Slightly_Open,smile]



    train_json_list.append(one_image_ann)





with open(train_json,'w') as f:
    json.dump(train_json_list, f,indent=2)





