import os
import torch
import argparse
from PIL import Image
from os.path import join
from utils.utils import *
from torch.utils.data import DataLoader
from loader.dataloader import ContrastiveSegDataset
from torchvision.transforms.functional import five_crop
from tqdm import tqdm
from torch.utils.data import Dataset

class RandomCropComputer(Dataset):

    def _get_size(self, img):
        if len(img.shape) == 3:
            return [int(img.shape[1] * self.crop_ratio), int(img.shape[2] * self.crop_ratio)]
        elif len(img.shape) == 2:
            return [int(img.shape[0] * self.crop_ratio), int(img.shape[1] * self.crop_ratio)]
        else:
            raise ValueError("Bad image shape {}".format(img.shape))

    def five_crops(self, i, img):
        return five_crop(img, self._get_size(img))

    def __init__(self, args, dataset_name, img_set, crop_type, crop_ratio):
        self.pytorch_data_dir = args.data_dir
        self.crop_ratio = crop_ratio

        if args.dataset=='coco171':
            self.save_dir = join(
                args.data_dir, 'cocostuff', "cropped", "coco171_{}_crop_{}".format(crop_type, crop_ratio))
        elif args.dataset=='coco81':
            self.save_dir = join(
                args.data_dir, 'cocostuff', "cropped", "coco81_{}_crop_{}".format(crop_type, crop_ratio))
        else:
            self.save_dir = join(
                args.data_dir, dataset_name, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        self.args = args

        self.img_dir = join(self.save_dir, "img", img_set)
        self.label_dir = join(self.save_dir, "label", img_set)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        # train dataset
        self.dataset = ContrastiveSegDataset(
            pytorch_data_dir=args.data_dir,
            dataset_name=args.dataset,
            crop_type=None,
            image_set=img_set,
            transform=get_transform(args.train_resolution, False, "center"),
            target_transform=get_transform(args.train_resolution, True, "center"),
            num_neighbors=7,
            extra_transform=lambda i, x: self.five_crops(i, x)
        )


    def __getitem__(self, item):
        batch = self.dataset[item]
        imgs = batch['img']
        labels = batch['label']
        for crop_num, (img, label) in enumerate(zip(imgs, labels)):
            img_num = item * 5 + crop_num
            img_arr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            label_arr = (label + 1).unsqueeze(0).permute(1, 2, 0).to('cpu', torch.uint8).numpy().squeeze(-1)
            Image.fromarray(img_arr).save(join(self.img_dir, "{}.jpg".format(img_num)), 'JPEG')
            Image.fromarray(label_arr).save(join(self.label_dir, "{}.png".format(img_num)), 'PNG')
        return True

    def __len__(self):
        return len(self.dataset)


def my_app():

    # fetch args
    parser = argparse.ArgumentParser()

    # fixed parameter
    parser.add_argument('--train_resolution', default=320, type=int)
    parser.add_argument('--num_workers', default=int(os.cpu_count() / 8), type=int)

    # dataset and baseline
    parser.add_argument('--data_dir', default='/mnt/hard2/lbk-iccv/datasets', type=str)
    parser.add_argument('--dataset', default='coco81', type=str)

    args = parser.parse_args()


    for img_set in ["train", "val"]:
        dataset = RandomCropComputer(args, args.dataset, img_set, "five", 0.5)
        loader = DataLoader(dataset, 1, shuffle=False, num_workers=args.num_workers, collate_fn=lambda l: l)
        for _ in tqdm(loader): pass


if __name__ == "__main__":
    my_app()