import numpy as np
import torch, torchvision
import os, logging, pickle, json, copy
import tqdm
import os.path as osp
import pdb
import clip
try:
    from . import data_utils
    from .. import config as cfg
except (ValueError, ImportError):
    import data_utils
    import sys
    sys.path.append("..")
    import config as cfg



class CompositionDatasetActivations(torch.utils.data.Dataset):

    def __init__(self, name, root, phase, feat_file=None, split='compositional-split', with_image=False, obj_pred=None):
        self.root = root
        self.phase = phase
        self.split = split
        self.with_image = with_image

        self.feat_dim = None
        self.transform = data_utils.imagenet_transform(phase)
        self.loader = data_utils.ImageLoader(self.root+'/images/')
        #pdb.set_trace()
        # read feature
        if feat_file is not None:
            feat_file = os.path.join(root, feat_file)
            activation_data = torch.load(feat_file)   # activation_data is a dictionary and it has two keys: features(a tensor), files(a list) 
            # pdb.set_trace()                           # features.shape [29126, 512]     len(files):29126
            self.activation_dict = dict(zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1) # 512
            print ('%d activations loaded'%(len(self.activation_dict)))
        
        # read pair info
        # pair = (attr, obj)
        self.attrs, self.objs, self.pairs, self.train_pairs, self.test_pairs = self.parse_split()
        assert len(set(self.train_pairs)&set(self.test_pairs))==0, 'train and test are not mutually exclusive'

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)} # 这里的attrs是训练+测试的所有attrs
        # 给每个属性加上序号  UT一共16个属性
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        # 给每个物体加上序号  UT一共12个物体
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        # 116
        # pdb.set_trace()

        self.train_data, self.test_data = self.get_split_info()
        self.data = self.train_data if self.phase=='train' else self.test_data   # list of [img_name, attr, obj, attr_id, obj_id, feat]
        print ('# images: %d'%len(self.data))
        print ('# train pairs: %d | # test pairs: %d'%(len(self.train_pairs), len(self.test_pairs)))
        # pdb.set_trace()
        # return {object: all attrs that occur with obj}
        self.obj_affordance_mask = []
        for _obj in self.objs:
            affordance = {sample["attr_name"] for sample in self.train_data+self.test_data if sample["obj_name"]==_obj}
            mask = [1 if x in affordance else 0 for x in self.attrs]
            self.obj_affordance_mask.append(mask)


        # negative image pool
        samples_grouped_by_obj = [[] for _ in range(len(self.objs))] # len=12
        # pdb.set_trace()
        for i,x in enumerate(self.train_data):   # len=24898
            samples_grouped_by_obj[x["obj_id"]].append(i) # len(samples_grouped_by_obj)=12 samples_grouped_by_obj里面有12个空，每次碰到对应的object就往里面加上编号,感觉是将traindata里面的所有数据做了分类

        self.neg_pool = []  # [obj_id][attr_id] => list of sample id
        # pdb.set_trace()
        for obj_id in range(len(self.objs)): # len=12
            self.neg_pool.append([])
            for attr_id in range(len(self.attrs)):
                self.neg_pool[obj_id].append(
                    [i for i in samples_grouped_by_obj[obj_id] if 
                        self.train_data[i]["attr_id"] != attr_id ]   # 将不属于train_data object的属性全部筛选出来
                )
        
        # len(self.neg_pool): 12
        # len(self.neg_pool[i]): 16
        # len(self.neg_pool[i][j]): 1607,1607,1607,1529......
        # pdb.set_trace()
        if obj_pred is None:
            self.obj_pred = None
        else:
            obj_pred_path = osp.join(cfg.DATA_ROOT_DIR, 'obj_scores', obj_pred)
            print("Loading object prediction from %s"%osp.basename(obj_pred_path))
            if obj_pred.endswith(".pkl"):
                # back compatible to TF version
                with open(obj_pred_path, 'rb') as fp:
                    self.obj_pred = np.array(pickle.load(fp), dtype=np.float32)
            else:
                self.obj_pred = torch.load(obj_pred_path)



    def get_split_info(self):

        data = torch.load(self.root+'/metadata.t7')
        # dict.keys() ['image','attr','obj','_image','set']
        # data[1]['image'] 'Leather_Shoes.Oxfords/100627.255.jpg'
        # data[1]['attr']   'Leather'
        # data[1]['obj']     'Shoes.Oxfords'
        # data[1]['_image']  'Shoes/Oxfords/Bostonian/100627.255.jpg'
        # data[1]['set']      'train'
        train_pair_set = set(self.train_pairs) # 集合
        test_pair_set = set(self.test_pairs)
        train_data, test_data = [], []
        #pdb.set_trace()
        for instance in data:

            image, attr, obj = instance['image'], instance['attr'], instance['obj']

            if attr=='NA' or (attr, obj) not in self.pairs:
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = {
                "image_path": image,
                "attr_name": attr,
                "obj_name": obj,
                "attr_id": self.attr2idx[attr],
                "obj_id": self.obj2idx[obj],
                "feature": self.activation_dict[image],
            }

            if (attr, obj) in train_pair_set:
                train_data.append(data_i)
            elif (attr, obj) in test_pair_set:
                test_data.append(data_i)
            else:
                raise ValueError("Invalid pair ({}, {})".format(attr, obj))
        # pdb.set_trace()
        return train_data, test_data  # 24898 4228
        # using 83 object-attribute pairs/24898 images as train set and 33 pairs/4228 images for testing

    def parse_split(self):

        def parse_pairs(pair_list):
            with open(pair_list,'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs('%s/%s/train_pairs.txt'%(self.root, self.split)) # 83
        ts_attrs, ts_objs, ts_pairs = parse_pairs('%s/%s/test_pairs.txt'%(self.root, self.split))  # 33
        # pdb.set_trace()
        all_attrs, all_objs = sorted(list(set(tr_attrs+ts_attrs))), sorted(list(set(tr_objs+ts_objs)))   # set()函数可以去除重复的元素 
        all_pairs = sorted(list(set(tr_pairs + ts_pairs)))  # 116

        return all_attrs, all_objs, all_pairs, tr_pairs, ts_pairs


    def sample_negative(self, attr_id, obj_id):
        return np.random.choice(self.neg_pool[obj_id][attr_id])      # 从不属于object的属性中随机筛选一个


    def __getitem__(self, index):
        def get_sample(i: int) -> dict:
            sample = copy.copy(self.data[i])  # UT: 24898
            if self.with_image:
                img = self.loader(sample["image_path"])
                img = self.transform(img)
                sample['image'] = img

            return sample
                # "image_path": image,
                # "image": img
                # "attr_name": attr,  'Leather'
                # "obj_name": obj,    'Shoes.Oxfords'
                # "attr_id": self.attr2idx[attr],  6
                # "obj_id": self.obj2idx[obj],     9
                # "feature": self.activation_dict[image],    [512] 512的image embedding
        pos = get_sample(index)
        
        mask = np.array(self.obj_affordance_mask[pos["obj_id"]], dtype=np.float32) # ？

        data = {
            "pos_attr_id":      pos["attr_id"],
            "pos_obj_id":       pos["obj_id"],
            "pos_feature":      pos["feature"],
            "affordance_mask":  mask,
        }

        if self.phase=='train':
            negid = self.sample_negative(pos["attr_id"], pos["obj_id"]) # negative example
            neg = get_sample(negid)
            data.update({
                "neg_attr_id":  neg["attr_id"],
                "neg_feature":  neg["feature"],
            })
        # pdb.set_trace()
        if self.obj_pred is not None:
            data["obj_pred"] = self.obj_pred[index,:]

        return data

    def __len__(self):
        return len(self.data)


# this code snippet is used to extract the features
class CompositionDatasetActivationsGenerator(CompositionDatasetActivations):

    def __init__(self, root, feat_file, split='compositional-split', feat_extractor=None):
        super(CompositionDatasetActivationsGenerator, self).__init__("aDataset", root, 'train', None, split)
        assert os.path.exists(root)
        with torch.no_grad():
            self.generate_features(feat_file, feat_extractor)
        print('Features generated.')
    # 这里需要重写一下这个函数     
    def get_split_info(self):
        data = torch.load(self.root+'/metadata.t7')
        #pdb.set_trace()
        train_pair_set = set(self.train_pairs)
        test_pair_set = set(self.test_pairs)
        train_data, test_data = [], []

        print("natural split")
        for instance in data:
            image, attr, obj = instance['image'], instance['attr'], instance['obj']

            if attr=='NA' or (attr, obj) not in self.pairs:
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue
                
            data_i = {
                "image_path": image,
                "attr_name": attr,
                "obj_name": obj,
                "attr_id": self.attr2idx[attr],
                "obj_id": self.obj2idx[obj],
                "feature": None,
            }

            if (attr, obj) in train_pair_set:
                train_data.append(data_i)
            elif (attr, obj) in test_pair_set:
                test_data.append(data_i)
            else:
                raise ValueError("Invalid pair ({}, {})".format(attr, obj))

        return train_data, test_data
    
    def generate_features(self, out_file, feat_extractor):

        data = self.train_data+self.test_data
        transform = data_utils.imagenet_transform('test')

        if feat_extractor is None:
            feat_extractor = torchvision.models.resnet18(pretrained=True)
            feat_extractor.fc = torch.nn.Sequential()
        feat_extractor.eval().cuda()
        
        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(data_utils.chunks(data, 512), total=len(data)//512):
            #pdb.set_trace()
            files = [x["image_path"] for x in chunk]   # MIT-states: len(file)=512
            imgs = list(map(self.loader, files))       # get 512 images  PIL.Image.Image
            imgs = list(map(transform, imgs))          # do some transforms on the image, and transform the image to tensors  imgs[i].shape:[3,224,224]
            with torch.no_grad():
                model, preprocess = clip.load("ViT-B/32") # ViT
                feats = model.encode_image(torch.stack(imgs, 0).cuda()).float()
            #pdb.set_trace()
            #feats = feat_extractor(torch.stack(imgs, 0).cuda()) # torch.stack(imgs, 0).shape: [512, 3, 224, 224]  feats.shape:[512, 512] 第一个是batchsize,第二个是向量维数
            image_feats.append(feats.data.cpu())       # len(image_feats):105
            image_files += files                       # len(image_files):53753
        #pdb.set_trace()
        image_feats = torch.cat(image_feats, 0)        # image_feats.shape:[53753, 512]
        print ('features for %d images generated'%(len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)
    


# {'Boots.Ankle': 0, 'Boots.Knee.High': 1, 'Boots.Mid-Calf': 2, 
# 'Sandals': 3, 'Shoes.Boat.Shoes': 4, 'Shoes.Clogs.and.Mules': 5, 
# 'Shoes.Flats': 6, 'Shoes.Heels': 7, 'Shoes.Loafers': 8, 'Shoes.Oxfords': 9, 
# 'Shoes.Sneakers.and.Athletic.Shoes': 10, 'Slippers': 11}

#{'Canvas': 0, 'Cotton': 1, 'Faux.Fur': 2, 'Faux.Leather': 3, 
# 'Full.grain.leather': 4, 'Hair.Calf': 5, 'Leather': 6, 'Nubuck': 7, 
# 'Nylon': 8, 'Patent.Leather': 9, 'Rubber': 10, 'Satin': 11, 
# 'Sheepskin': 12, 'Suede': 13, 'Synthetic': 14, 'Wool': 15}

# Canvas flat shoes self.pos_pool[6][0]   93张图片  neg是cotton attr_id是1
if __name__=='__main__':
    """example code for generating new features for MIT states and UT Zappos
    CompositionDatasetActivationsGenerator(
        root = 'data-dir', 
        feat_file = 'filename-to-save', 
        feat_extractor = torchvision.models.resnet18(pretrained=True),
    )
    """
    CompositionDatasetActivationsGenerator(
        root = '/home/user/lmy/SymNet_torch_dev/data2/mit-states',
        feat_file = '/home/user/lmy/SymNet_torch_dev/data2/mit-states/mit-feature.t7',
    )










