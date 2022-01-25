from __future__ import print_function

from dcsgan.miscc.config import cfg, cfg_from_file
import dcsgan.pororo_data as data
import PIL
import functools
from tqdm import tqdm

import os
import sys
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import models

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ground_truth', action='store_true')
    parser.add_argument('--img_dir', type=str, default='')
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()
    return args

def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)

# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, drop_prob=0.2,
                 nhidden=256, nlayers=1, bidirectional=True, video_len=5, rnn_type='lstm'):
        super(RNN_ENCODER, self).__init__()

        self.vocab_size = vocab_size  # size of the dictionary
        self.emb_dim = emb_dim  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions
        self.video_len = video_len

        self.define_module()
        # self.init_weights()

    def define_module(self):
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'lstm':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.word_rnn = nn.LSTM(self.emb_dim, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
            self.sent_rnn = nn.LSTM(self.nhidden*self.num_directions, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'gru':
            self.word_rnn = nn.GRU(self.emb_dim, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
            self.sent_rnn = nn.GRU(self.nhidden*self.num_directions, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz, cuda):
        if self.rnn_type == 'lstm':
            if cuda:
                return (Variable(torch.zeros(self.nlayers * self.num_directions, bsz*self.video_len, self.nhidden)).cuda(),
                        Variable(torch.zeros(self.nlayers * self.num_directions, bsz*self.video_len, self.nhidden)).cuda()), \
                       (Variable(torch.zeros(self.nlayers * self.num_directions, bsz, self.nhidden)).cuda(),
                        Variable(torch.zeros(self.nlayers * self.num_directions, bsz, self.nhidden)).cuda())
            else:
                return (Variable(torch.zeros(self.nlayers * self.num_directions, bsz*self.video_len, self.nhidden)),
                        Variable(torch.zeros(self.nlayers * self.num_directions, bsz*self.video_len, self.nhidden))), \
                       (Variable(torch.zeros(self.nlayers * self.num_directions, bsz, self.nhidden)),
                        Variable(torch.zeros(self.nlayers * self.num_directions, bsz, self.nhidden)))
        else:
            if cuda:
                return Variable(torch.zeros(self.nlayers * self.num_directions, bsz*self.video_len, self.nhidden).cuda()), \
                       Variable(torch.zeros(self.nlayers * self.num_directions, bsz, self.nhidden).cuda())
            else:
                return Variable(torch.zeros(self.nlayers * self.num_directions,
                                       bsz, self.nhidden)), Variable(torch.zeros(self.nlayers * self.num_directions,
                                       bsz, self.nhidden))

    def forward(self, captions, cap_lens, word_hidden, sent_hidden):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.embeddings(captions))
        # print(emb.shape, emb.type(), word_hidden[0].type(), word_hidden[1].type(), emb.dtype, word_hidden[0].dtype, word_hidden[1].dtype)
        # print(emb.shape, emb.type(), word_hidden.type(), emb.dtype, word_hidden.dtype)
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        # print(cap_lens)
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True, enforce_sorted=False)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        # print(emb.data.type(), emb.data.dtype)
        # print('Performing forward pass on word RNN')
        output, hidden = self.word_rnn(emb, word_hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # print(output.shape)
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # print(words_emb.shape)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'lstm':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()

        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)

        sent_rnn_input = sent_emb.view(-1, self.video_len, self.nhidden * self.num_directions)
        # print(sent_rnn_input.shape)
        # print('Performing forward pass on sentence RNN')
        _, story_hidden = self.sent_rnn(sent_rnn_input, sent_hidden)
        if self.rnn_type == 'lstm':
            story_emb = story_hidden[0].transpose(0, 1).contiguous()
        else:
            story_emb = story_hidden.transpose(0, 1).contiguous()
        story_emb = story_emb.view(-1, self.nhidden * self.num_directions)
        # print(story_emb.shape)

        return words_emb, sent_emb, story_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        num_classes = 9
        model_ft = models.inception_v3(pretrained=False)
        for param in model_ft.parameters():
            param.requires_grad = False

        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # model_ft.load_state_dict(
        #     torch.load('/ssd-playpen/home/adyasha/projects/StoryGAN/classifier/old_models/epoch-49.pt'))

        # for param in model_ft.parameters():
        #     param.requires_grad = False
        # print('Loaded pretrained model from  /ssd-playpen/home/adyasha/projects/StoryGAN/classifier/old_models/epoch-49.pt')
        # print(model)

        self.define_module(model_ft)
        # self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)
        self.story_transform = nn.Linear(self.nef, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)
        self.story_transform.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):

        video_len = x.shape[1]
        # print(x.shape)
        x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])

        # --> fixed-size input: batch x 3 x 299 x 299
        # x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        features = self.emb_features(features)
        story_features = self.story_transform(torch.mean(cnn_code.view(-1, video_len, self.nef), dim=1).view(-1, self.nef))

        return features, cnn_code, story_features

def evaluate(dataloader, cnn_model, rnn_model, batch_size, debug=False, n_trials=10):

    if not debug:
        cnn_model.eval()

    all_sent_code = []
    all_story_code = []
    all_sent_emb = []
    all_story_emb = []

    rnn_model.eval()
    for step, st_batch in tqdm(enumerate(dataloader, 0)):

        st_imgs = st_batch['images'].transpose(1, 2)
        st_input_ids = Variable(st_batch['input_ids'])
        st_masks = Variable(st_batch['masks'])
        video_len = st_input_ids.shape[1]

        if cfg.CUDA:
            st_imgs = st_imgs.cuda()
            st_input_ids = st_input_ids.cuda()
            st_masks = st_masks.cuda()

        if not debug:
            _, sent_code, story_code = cnn_model(st_imgs)
        else:
            batch_size = st_imgs.shape[0]
            sent_code = torch.zeros(batch_size*5, 256).cuda()
            story_code = torch.zeros(batch_size, 256).cuda()

        word_hidden, sent_hidden = rnn_model.init_hidden(batch_size, cfg.CUDA)
        caption_lens = torch.sum(st_masks, dim=-1).view(-1)
        _, sent_emb, story_emb = rnn_model(st_input_ids.view(-1, st_input_ids.shape[-1]), caption_lens,
                                                   word_hidden, sent_hidden)

        first_img_idxs = list(range(story_emb.shape[0]))*5

        # print(sent_code.shape, story_code.shape, sent_emb.shape, story_emb.shape)
        all_sent_code.append(sent_code[first_img_idxs, :])
        all_story_code.append(story_code)
        all_sent_emb.append(sent_emb[first_img_idxs, :])
        all_story_emb.append(story_emb)

        if debug:
            if step == 20:
                break

    all_sent_code = torch.cat(all_sent_code, dim=0)
    all_story_code = torch.cat(all_story_code, dim=0)
    all_sent_emb = torch.cat(all_sent_emb, dim=0)
    all_story_emb = torch.cat(all_story_emb, dim=0)
    print(all_sent_emb.shape, all_story_emb.shape, all_sent_code.shape, all_story_code.shape)
    n_samples = all_story_emb.shape[0]

    # Story R-Precision
    cosine_sim_fn = nn.CosineSimilarity(dim=-1)
    sample_idxs = list(range(n_samples))
    p_rates = []
    for n in range(n_trials):
        p_rate = []
        for i in range(n_samples):
            img_code = all_story_code[i].repeat(100, 1)
            idxs = random.sample(sample_idxs[:i] + sample_idxs[i+1:], k=99) + [i]
            story_embs = all_story_emb[idxs, :]
            assert story_embs.shape[0] == 100
            img_cos = cosine_sim_fn(img_code, story_embs)
            # print(img_cos)
            _, indices = torch.sort(img_cos, descending=True)
            top = indices[0]
            # print(img_cos.shape, indices)
            # print(top)
            if top.data == 99:
                p_rate.append(1)
            else:
                p_rate.append(0)
        p_rate = sum(p_rate)/n_samples
        print("Trial %s/%s: R-Precision: %s" % (n, n_trials, p_rate))
        p_rates.append(p_rate)

    print("R-Precision: %s +/- %s" % (np.mean(p_rates), np.std(p_rates)))


def load_models(model_dir, vocab_size, emb_dim, hidden_dim, debug=False):
    # build model ############################################################
    text_encoder = RNN_ENCODER(vocab_size, emb_dim=emb_dim, nhidden=hidden_dim)

    if debug:
        image_encoder = None
    else:
        image_encoder = CNN_ENCODER(hidden_dim)

    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict =  torch.load(os.path.join(model_dir, cfg.TRAIN.NET_E))
        text_encoder.load_state_dict(state_dict)
        print('Loading text encoder weights from ', os.path.join(model_dir, cfg.TRAIN.NET_E))

        if not debug:
            name = os.path.join(model_dir, cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder'))
            state_dict = torch.load(name)
            image_encoder.load_state_dict(state_dict)
            print('Loading image encoder weights from ', name)

    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        if not debug:
            image_encoder = image_encoder.cuda()

    return text_encoder, image_encoder


def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))
    vid = torch.stack(vid).permute(1, 0, 2, 3)
    return vid

if __name__ == "__main__":

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    dir_path = cfg.DATA_DIR

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    num_gpu = 1
    print('CUDA is ', cfg.CUDA)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    # timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    # output_dir = './output/%s_%s' % \
    #     (cfg.DATASET_NAME, cfg.CONFIG_NAME)

    torch.cuda.set_device(0)
    cudnn.benchmark = True

    im_input_size = 299
    transform_val = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Resize(im_input_size),
        transforms.CenterCrop(im_input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    video_transform_val = functools.partial(video_transform, image_transform=transform_val)

    counter = np.load(os.path.join(dir_path, 'frames_counter.npy'), allow_pickle=True).item()
    print("----------------------------------------------------------------------------------")
    print("Preparing Testing dataset")

    # out_dir = '/ssd-playpen/home/adyasha/projects/StoryGAN/src/output/pororo_transformer_stageI_r1.0/Test/images-epoch-150/'
    out_dir = args.img_dir

    base_test = data.VideoFolderDataset(dir_path, counter, dir_path, 4, args.mode)
    testdataset = data.StoryDataset(base_test, dir_path, video_transform_val, return_caption=True, out_dir=out_dir)
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=cfg.TRAIN.ST_BATCH_SIZE,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

    n_hidden = 256
    # Train ##############################################################
    text_encoder, image_encoder = load_models(args.output_dir, len(testdataset.vocab), cfg.TEXT.DIMENSION, hidden_dim=n_hidden, debug=args.debug)
    para = list(text_encoder.parameters())
    if not args.debug:
        for v in image_encoder.parameters():
            if v.requires_grad:
                para.append(v)

    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        eval_batch_size = cfg.TRAIN.ST_BATCH_SIZE
        evaluate(testloader, image_encoder, text_encoder, eval_batch_size, args.debug)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')