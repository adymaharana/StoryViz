from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import PIL
import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import numpy as np
import functools
import dcsgan.pororo_data as data

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from dcsgan.miscc.config import cfg, cfg_from_file
from dcsgan.trainer import GANTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file', help='config file', required=True, type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, required=True)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--output_root_dir', type=str, default='./out/')

    #### inference args
    parser.add_argument('--checkpoint', type=str, default='', help='path to trained checkpoint')
    parser.add_argument('--infer_dir', type=str, default='', help='path to output directory')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    dir_path = args.data_dir

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    print('Using config:')
    pprint.pprint(cfg)

    if cfg.USE_MART:
        assert not cfg.USE_TRANSFORMER, "Either choose Transformer (Non-recurrent) or Memory Augmented Recurrent Transformer (MART)"
    if cfg.USE_TRANSFORMER:
        assert not cfg.USE_MART, "Either choose Transformer (Non-recurrent) or Memory Augmented Recurrent Transformer (MART)"

    random.seed(0)
    torch.manual_seed(0)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(0)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(args.output_root_dir, cfg.CONFIG_NAME)

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        image_transforms = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((cfg.IMSIZE, cfg.IMSIZE)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # dataset = TextDataset(cfg.DATA_DIR, 'train',
        #                       imsize=cfg.IMSIZE,
        #                       transform=image_transform)
        #assert dataset
        def video_transform(video, image_transform):
            vid = []
            for im in video:
                vid.append(image_transform(im))
            vid = torch.stack(vid).permute(1, 0, 2, 3)
            return vid

        video_len = 5
        n_channels = 3
        video_transforms = functools.partial(video_transform, image_transform=image_transforms)

        counter = np.load(os.path.join(dir_path, 'frames_counter.npy'), allow_pickle=True).item()
        print("----------------------------------------------------------------------------------")
        print("Preparing TRAINING dataset")
        base = data.VideoFolderDataset(dir_path, counter = counter, cache = dir_path, min_len = 4, mode="train")
        storydataset = data.StoryDataset(base, dir_path, video_transforms,
                                         return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL)
        if cfg.USE_MART:
            if cfg.MART.vocab_glove_path == '':
                cfg.MART.vocab_glove_path = os.path.join(dir_path, 'mart_glove_embeddings.mat')
            storydataset.vocab.extract_glove(cfg.MART.raw_glove_path, cfg.MART.vocab_glove_path)
            cfg.MART.pretrained_embeddings = cfg.MART.vocab_glove_path

        imagedataset = data.ImageDataset(base, dir_path, image_transforms,
                                         return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL)

        imageloader = torch.utils.data.DataLoader(
            imagedataset, batch_size=cfg.TRAIN.IM_BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        storyloader = torch.utils.data.DataLoader(
            storydataset, batch_size=cfg.TRAIN.ST_BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        print("----------------------------------------------------------------------------------")
        print("Preparing EVALUATION dataset")

        test_dir_path = dir_path
        base_test = data.VideoFolderDataset(test_dir_path, counter, test_dir_path, 4, mode="val")
        testdataset = data.StoryDataset(base_test, test_dir_path, video_transforms, return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL)
        testloader = torch.utils.data.DataLoader(
            testdataset, batch_size=20,
            drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

        # update vocab config parameters for MART
        if cfg.USE_MART:
            cfg.MART.vocab_size = len(storydataset.vocab)
            cfg.MART.max_t_len = storydataset.max_len
            cfg.MART.max_position_embeddings = storydataset.max_len

        if cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL:
            cfg.VOCAB_SIZE = len(storydataset.vocab)

        cfg.DATASET_NAME = 'pororo'

        algo = GANTrainer(cfg, output_dir, ratio = 1.0)
        algo.train(imageloader, storyloader, testloader, cfg.STAGE)
    else:

        assert args.checkpoint
        assert args.infer_dir

        if not os.path.exists(args.infer_dir):
            os.makedirs(args.infer_dir)

        def video_transform(video, image_transform):
            vid = []
            for im in video:
                vid.append(image_transform(im))
            vid = torch.stack(vid).permute(1, 0, 2, 3)
            return vid

        image_transforms = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((cfg.IMSIZE, cfg.IMSIZE)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        video_transforms = functools.partial(video_transform, image_transform=image_transforms)
        counter = np.load(os.path.join(dir_path, 'frames_counter.npy'), allow_pickle=True).item()

        test_dir_path = dir_path
        base_test = data.VideoFolderDataset(test_dir_path, counter, test_dir_path, 4, mode='val')
        testdataset = data.StoryDataset(base_test, test_dir_path, video_transforms,
                                        return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL)

        if cfg.USE_MART:
            testdataset.init_mart_vocab()
            print("Built vocabulary of %s words" % len(testdataset.vocab))
            if cfg.MART.vocab_glove_path == '':
                cfg.MART.vocab_glove_path = os.path.join(dir_path, 'martgan_embeddings.mat')
            testdataset.vocab.extract_glove(cfg.MART.raw_glove_path, cfg.MART.vocab_glove_path)
            cfg.MART.pretrained_embeddings = cfg.MART.vocab_glove_path

        # update vocab config parameters for MART
        if cfg.USE_MART:
            cfg.MART.vocab_size = len(testdataset.vocab)
            cfg.MART.max_t_len = testdataset.max_len
            cfg.MART.max_position_embeddings = testdataset.max_len

        if cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL:
            cfg.VOCAB_SIZE = len(testdataset.vocab)

        testloader = torch.utils.data.DataLoader(
            testdataset, batch_size=4,
            drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

        algo = GANTrainer(cfg)
        algo.sample(testloader, args.checkpoint, args.infer_dir, cfg.STAGE)