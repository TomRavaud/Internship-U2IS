# preparation de l'environnement

from __future__ import division, print_function

import os
import warnings  # To ignore warnings


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

warnings.filterwarnings("ignore")

DATASET = "datasets/dataset_sample_bag/"
# DATASET = "datasets/dataset_rania_2022-07-01-11-40-52/"


class DataProject(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):

        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.data_frame.iloc[idx, 1:]  # type numpy.float64
        landmarks = np.array(landmarks)

        landmarks = float(landmarks)

        sample = {'image': image, 'landmarks': landmarks}
        # hne 5ater ye5dedh dict

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    # pour le test de size
    def affichage(self):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[5, 0])
        image = io.imread(img_name)
        tupleee = image.shape
        print(tupleee)

        print(type(image))


# Data vizualization

def data_plot(dataproject, cols, rows):

    dataproject.affichage()
    #figure = plt.figure(figsize=(200, 200))
    # figure2=plt.figure()
    for i in range(1, cols * rows + 1):
        #sample_idx = torch.randint(len(dataproject), size=(1,)).item()
        plt.figure(1)
        plt.subplot(rows, cols, i)
        sample = dataproject[i]
        #figure.add_subplot(rows, cols, i)

        # plt.tight_layout()
        plt.title(sample['landmarks'])
        plt.subplots_adjust(wspace=1, hspace=1)
        plt.axis("off")
        plt.imshow(sample['image'].squeeze())

        plt.figure(2)
        plt.subplot(rows, cols, i)
        plt.hist(np.array(sample['image']).ravel(), bins=50, density=True)
        plt.xlabel("pixel values")
        plt.ylabel("relative frequency")
        plt.title("distribution of pixels")
        plt.subplots_adjust(wspace=1, hspace=1)

    plt.savefig("sample_training_data.png")
    plt.close()
   # loading data


# rescaling image's size

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        #image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        # if h > w:
        #new_h, new_w = self.output_size * h / w, self.output_size
        # else:
        #new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #landmarks = landmarks * [new_w / w, new_h / h]

        # return {'image': img, 'landmarks': landmarks}
        return img


class ToTensor(object):

    def __call__(self, image):
        #image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        # (a voir si'il y aura un problème)
        imagee = torch.from_numpy(image)
        imageee = torch.Tensor.float(imagee)
        # return {'image': imageee,
        # 'landmarks': landmarks}
        return imagee


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        #image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        #left = np.random.randint(0, w/2 - new_w/2)

        # image = image[top: top + new_h,
        # left: left + new_w]

        image = image[top:top+new_h,
                      int(w/2)-int(new_w/2):int(w/2)+int(new_w/2)]

        # return {'image': image, 'landmarks': landmarks}
        return image


def calcul_moy_std(training_data):

    sumr = 0
    sumg = 0
    sumb = 0
    stdr = 0
    stdb = 0
    stdg = 0
    loader = DataLoader(training_data, batch_size=len(
        training_data), num_workers=8)
    data = next(iter(loader))
    m = (data['image'][0]).numpy()
    # je suis pas sur de mean axis

    h, w, c = m.shape
    for i in data['image']:
        m = i.numpy()
        r = m[:, :, 0]
        g = m[:, :, 1]
        b = m[:, :, 2]
        sumr = sumr+np.sum(r)
        sumg = sumg+np.sum(g)
        sumb = sumb+np.sum(b)
    # print(h,w)

    meanr = sumr/(h*w*len(training_data))
    meang = sumg/(h*w*len(training_data))
    meanb = sumb/(h*w*len(training_data))
    mean = (meanr, meang, meanb)
    # print('mean')
    # print(meanr,meang,meanb)
    for i in data['image']:
        m = i.numpy()
        r = m[:, :, 0]
        g = m[:, :, 1]
        b = m[:, :, 2]
        stdr = stdr+np.sum(((r - meanr)**2))
        stdg = stdg+np.sum(((g - meang)**2))
        stdb = stdb+np.sum(((b - meanb)**2))
    stddr = np.sqrt(stdr/(h*w*len(training_data)))
    stddb = np.sqrt(stdb/(h*w*len(training_data)))
    stddg = np.sqrt(stdg/(h*w*len(training_data)))
    std = (stddr, stddg, stddb)

    return (mean, std)


class Normalize(object):

    def __init__(self, mean, std):
        self.std = std
        self.mean = mean

    def __call__(self, sample):

        # tr=transforms.Normalize(self.mean,self.std)
        # return tr(sample['image'])
        sample['image'][:, :, 0] = (
            sample['image'][:, :, 0]-self.mean[0])/self.std[0]
        sample['image'][:, :, 1] = (
            sample['image'][:, :, 1]-self.mean[1])/self.std[1]
        sample['image'][:, :, 2] = (
            sample['image'][:, :, 2]-self.mean[2])/self.std[2]
        return sample


# loading DataSet after transformation


# t=calcul_moy_std(training_data)
#print('le coue',t)


# imgs = torch.stack([training_data[i]['image']
#                    for i in range(len(training_data))], dim=3)
# # print(len(imgs))
# # print(imgs[0])
# print('ici', imgs.shape)
# mean = tuple(torch.Tensor.float(imgs.view(3, -1)).mean(dim=1).tolist())
# ecart = tuple(torch.Tensor.float(imgs.view(3, -1)).std(dim=1).tolist())
# print(mean, ecart)
# t = (mean, ecart)

data_transforms1 = transforms.Compose([


    # Apply histogram equalization
    Rescale(280),
    RandomCrop(225),
    ToTensor(),
    # transforms.Normalize(t[0],t[1]),


    # Add channel dimension to be able to apply convolutions

])


training_data1 = DataProject(
    csv_file=DATASET+'/imu_data.csv', root_dir=DATASET+'/zed_node_rgb_image_rect_color', transform=data_transforms1
)


# loading data in an iterator dataloader
# shuffle= True pour le choix aleatoire


# visualizing a batch

def show_landmarks_batch(sample_batched):

    images_batch, landmarks_batch = \
        sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))  # from tensor to ndarray

    # for i in range(batch_size):

    #plt.title('Batch from dataloader')


def load(training_data1, batch_size):
    L = len(training_data1)

    # un choix

    train_count = int(0.8 * L)

    valid_count = int(0.1 * L)

    test_count = L-valid_count-train_count
    #train_loader = DataLoader(training_data1, batch_size=Batch_size,shuffle=True, num_workers=4)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        training_data1, [train_count, valid_count, test_count])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    dataloaders = {'train': train_loader,
                   'valid': valid_loader, 'test': test_loader}

    for i_batch, sample_batched in enumerate(train_loader):

        # print(i_batch, sample_batched['image'].size(),
        # sample_batched['landmarks'].size())

        if i_batch == 3:
            plt.figure(1)
            show_landmarks_batch(sample_batched)
            #print(sample_batched ['image'])

            #print('hh', sample_batched['image'][1].size())

            plt.title((sample_batched['landmarks']).tolist())
            plt.axis('off')
            plt.ioff()
            #plt.show()
            plt.figure(2)
            grid = utils.make_grid(sample_batched['image'][0])
            k = grid.numpy().transpose((1, 2, 0))
            # k=np.array(sample_batched ['image'][0]).transpose((1, 2, 0)
            print(k.shape)
            plt.hist(k.ravel(), bins=50, density=True)
            plt.xlabel("pixel values")
            plt.ylabel("relative frequency")
            plt.title("distribution of pixels")
            # plt.show()
            break
    return dataloaders

