import random
from os.path import join

import torch.multiprocessing
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.datasets import VOCSegmentation


from utils.utils import *

from torch.utils.data.distributed import DistributedSampler


def dataloader(args, no_ddp_train_shuffle=True):

    if args.dataset == "cocostuff27":
        args.n_classes = 27
    elif args.dataset == "cityscapes":
        args.n_classes = 27
    elif args.dataset == "pascalvoc":
        args.n_classes = 21
    elif args.dataset == "coco80":
        args.n_classes = 80
    elif args.dataset == "coco171":
        args.n_classes = 171

    # train dataset
    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=args.data_dir,
        dataset_name=args.dataset,
        crop_type="five",
        image_set="train",
        transform=get_transform(args.train_resolution, False, "center"),
        target_transform=get_transform(args.train_resolution, True, "center"),
        num_neighbors=7,
        mask=True,
        pos_images=True,
        pos_labels=True
    )

    if args.distributed: train_sampler = DistributedSampler(train_dataset, shuffle=True)

    # train loader
    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=False if args.distributed else no_ddp_train_shuffle, num_workers=args.num_workers,
                              pin_memory=True, sampler=train_sampler if args.distributed else None)

    test_dataset = ContrastiveSegDataset(
        pytorch_data_dir=args.data_dir,
        dataset_name=args.dataset,
        crop_type=None,
        image_set="val",
        transform=get_transform(args.test_resolution, False, "center"),
        target_transform=get_transform(args.test_resolution, True, "center"),
        mask=True,
    )

    if args.distributed: test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # test dataloader
    test_loader = DataLoader(test_dataset, args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=False, sampler=test_sampler if args.distributed else None)

    sampler = train_sampler if args.distributed else None

    return train_loader, test_loader, sampler


class Coco(Dataset):
    def __init__(self, root, image_set, transform, target_transform,
                 coarse_labels, exclude_things, subset=None):
        super(Coco, self).__init__()
        self.split = image_set
        self.root = join(root, "cocostuff")
        self.coarse_labels = coarse_labels
        self.transform = transform
        self.label_transform = target_transform
        self.subset = subset
        self.exclude_things = exclude_things

        if self.subset is None:
            self.image_list = "Coco164kFull_Stuff_Coarse.txt"
        elif self.subset == 6:  # IIC Coarse
            self.image_list = "Coco164kFew_Stuff_6.txt"
        elif self.subset == 7:  # IIC Fine
            self.image_list = "Coco164kFull_Stuff_Coarse_7.txt"

        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }

        self.image_files = []
        self.label_files = []
        for split_dir in split_dirs[self.split]:
            with open(join(self.root, "curated", split_dir, self.image_list), "r") as f:
                img_ids = [fn.rstrip() for fn in f.readlines()]
                for img_id in img_ids:
                    self.image_files.append(join(self.root, "images", split_dir, img_id + ".jpg"))
                    self.label_files.append(join(self.root, "annotations", split_dir, img_id + ".png"))

        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

        self._label_names = [
            "ground-stuff",
            "plant-stuff",
            "sky-stuff",
        ]
        self.cocostuff3_coarse_classes = [23, 22, 21]
        self.first_stuff_index = 12

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.label_transform(Image.open(label_path)).squeeze(0)
        label[label == 255] = -1  # to be consistent with 10k
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1

        if self.coarse_labels:
            coarser_labels = -torch.ones_like(label)
            for i, c in enumerate(self.cocostuff3_coarse_classes):
                coarser_labels[coarse_label == c] = i
            return img, coarser_labels, coarser_labels >= 0
        else:
            if self.exclude_things:
                return img, coarse_label - self.first_stuff_index, (coarse_label >= self.first_stuff_index)
            else:
                return img, coarse_label, coarse_label >= 0

    def __len__(self):
        return len(self.image_files)


class CityscapesSeg(Dataset):
    def __init__(self, root, image_set, transform, target_transform):
        super(CityscapesSeg, self).__init__()
        self.split = image_set
        self.root = join(root, "cityscapes")
        if image_set == "train":
            # our_image_set = "train_extra"
            # mode = "coarse"
            our_image_set = "train"
            mode = "fine"
        else:
            our_image_set = image_set
            mode = "fine"
        self.inner_loader = Cityscapes(self.root, our_image_set,
                                       mode=mode,
                                       target_type="semantic",
                                       transform=None,
                                       target_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        self.first_nonvoid = 7

    def __getitem__(self, index):
        if self.transform is not None:
            image, target = self.inner_loader[index]

            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.target_transform(target)

            target = target - self.first_nonvoid
            target[target < 0] = -1
            mask = target == -1
            return image, target.squeeze(0), mask
        else:
            return self.inner_loader[index]

    def __len__(self):
        return len(self.inner_loader)


class CroppedDataset(Dataset):
    def __init__(self, root, dataset_name, crop_type, crop_ratio, image_set, transform, target_transform):
        super(CroppedDataset, self).__init__()
        self.dataset_name = dataset_name
        self.split = image_set
        self.root = join(root, dataset_name, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = join(self.root, "img", self.split)
        self.label_dir = join(self.root, "label", self.split)
        self.num_images = len(os.listdir(self.img_dir))
        assert self.num_images == len(os.listdir(self.label_dir))

    def __getitem__(self, index):
        image = Image.open(join(self.img_dir, "{}.jpg".format(index))).convert('RGB')
        target = Image.open(join(self.label_dir, "{}.png".format(index))) if index!=23385 else 0

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        random.seed(seed)
        torch.manual_seed(seed)
        target = self.target_transform(target) if index!=23385 else torch.zeros([1, image.shape[1], image.shape[2]], dtype=torch.int64)

        target = target - 1
        mask = target == -1
        return image, target.squeeze(0), mask

    def __len__(self):
        return self.num_images

class PascalVOC(VOCSegmentation):
    def __init__(self, root, year, image_set, download, transforms, target_transforms):
        super().__init__(root, year=year, image_set=image_set, download=download, transforms=transforms)
        self.target_transforms=target_transforms
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.masks[idx])

        if self.transforms is not None:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transforms(image)
            self._set_seed(seed); label = self.target_transforms(label)
            label[label > 20] = -1
        return image, label
    
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def PascalVOCGenerator(root, image_set, transform, target_transform):
        return PascalVOC(join(root, "pascalvoc"), 
                    year='2012', 
                    image_set=image_set, 
                    download=False, 
                    transforms=transform,
                    target_transforms=target_transform)

class ContrastiveSegDataset(Dataset):
    def __init__(self,
                 pytorch_data_dir,
                 dataset_name,
                 crop_type,
                 image_set,
                 transform,
                 target_transform,
                 aug_geometric_transform=None,
                 aug_photometric_transform=None,
                 num_neighbors=5,
                 mask=False,
                 pos_labels=False,
                 pos_images=False,
                 extra_transform=None,
                 ):
        super(ContrastiveSegDataset).__init__()
        self.num_neighbors = num_neighbors
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.mask = mask
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.extra_transform = extra_transform
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if  dataset_name == "cityscapes" and crop_type is None:
            self.n_classes = 27
            dataset_class = CityscapesSeg
            extra_args = dict()
        elif dataset_name == "cityscapes" and crop_type is not None:
            self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cityscapes", crop_type=crop_type, crop_ratio=0.5)
        elif dataset_name == "cocostuff3":
            self.n_classes = 3
            dataset_class = Coco
            extra_args = dict(coarse_labels=True, subset=6, exclude_things=True)
        elif dataset_name == "cocostuff15":
            self.n_classes = 15
            dataset_class = Coco
            extra_args = dict(coarse_labels=False, subset=7, exclude_things=True)
        elif dataset_name == "cocostuff27" and crop_type is not None:
            self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cocostuff", crop_type="five", crop_ratio=0.5)
        elif dataset_name == "cocostuff27" and crop_type is None:
            self.n_classes = 27
            dataset_class = Coco
            extra_args = dict(coarse_labels=False, subset=None, exclude_things=False)
            if image_set == "val":
                extra_args["subset"] = 7
        elif dataset_name == "pascalvoc":
            self.n_classes = 21
            dataset_class = PascalVOC.PascalVOCGenerator
            extra_args = dict()
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = dataset_class(
            root=pytorch_data_dir,
            image_set=self.image_set,
            transform=transform,
            target_transform=target_transform, **extra_args)

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, ind):
        pack = self.dataset[ind]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        self._set_seed(seed)
        if self.extra_transform is not None:
            extra_trans = self.extra_transform
        else:
            extra_trans = lambda i, x: x

        ret = {
            "ind": ind,
            "img": self.normalize(extra_trans(ind, pack[0])),
            "label": extra_trans(ind, pack[1]),
        }

        return ret

