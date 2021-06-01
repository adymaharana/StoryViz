from __future__ import print_function

from dcsgan.miscc.utils import mkdir_p
from dcsgan.miscc.utils import sent_loss, words_loss
from dcsgan.miscc.config import cfg, cfg_from_file
import dcsgan.pororo_data as data
import PIL
import functools
from tqdm import tqdm


import os
import sys
import time
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
import torch.optim as optim
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
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--debug', action='store_true')
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
        model_ft = models.inception_v3(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False

        for param in model_ft.parameters():
            param.requires_grad = False


        self.define_module(model_ft)
        self.init_trainable_weights()

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


def train(dataloader, cnn_model, rnn_model, batch_size, optimizer, epoch, debug=False):

    if not debug:
        cnn_model.train()
    rnn_model.train()
    st_total_loss0 = 0
    st_total_loss1 = 0
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    for step, st_batch in tqdm(enumerate(dataloader, 0)):
        # print('step', step)
        rnn_model.zero_grad()
        if not debug:
            cnn_model.zero_grad()

        st_imgs = st_batch['images'].transpose(1, 2)
        st_input_ids = Variable(st_batch['input_ids'])
        st_masks = Variable(st_batch['masks'])

        video_len = st_input_ids.shape[1]


        if cfg.CUDA:
            st_imgs = st_imgs.cuda()
            st_input_ids = st_input_ids.cuda()
            st_masks = st_masks.cuda()


        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        if not debug:
            words_features, sent_code, story_code = cnn_model(st_imgs)
        else:
            batch_size = st_imgs.shape[0]
            words_features = torch.zeros(batch_size*5, 256, 17, 17).cuda()
            sent_code = torch.zeros(batch_size*5, 256).cuda()
            story_code = torch.zeros(batch_size, 256).cuda()

        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        word_hidden, sent_hidden = rnn_model.init_hidden(batch_size, cfg.CUDA)
        # print(word_hidden[0].shape, word_hidden[1].shape, sent_hidden[0].shape, sent_hidden[1].shape, word_hidden[0].type(), sent_hidden[0].type())
        # print(word_hidden.shape, sent_hidden.shape, word_hidden.type(), sent_hidden.type(), word_hidden.dtype)

        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        caption_lens = torch.sum(st_masks, dim=-1).view(-1)
        words_emb, sent_emb, story_emb = rnn_model(st_input_ids.view(-1, st_input_ids.shape[-1]), caption_lens,
                                                   word_hidden, sent_hidden)
        # print(words_emb.shape, words_features.shape)
        # print(words_features.shape, words_emb.shape)
        sent_labels = Variable(torch.LongTensor(range(cfg.TRAIN.ST_BATCH_SIZE*video_len)))
        if cfg.CUDA:
            sent_labels = sent_labels.cuda()
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, sent_labels,
                                                 caption_lens, batch_size*video_len)

        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1


        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, sent_labels, cfg.TRAIN.SMOOTH.GAMMA3)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data

        story_labels = Variable(torch.LongTensor(range(cfg.TRAIN.ST_BATCH_SIZE)))
        if cfg.CUDA:
            story_labels = story_labels.cuda()
        st_loss0, st_loss1 = sent_loss(story_code, story_emb, story_labels, cfg.TRAIN.SMOOTH.GAMMA4)
        loss += st_loss0 + st_loss1
        st_total_loss0 += st_loss0.data
        st_total_loss1 += st_loss1.data

        #
        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0 / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1 / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0 / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1 / UPDATE_INTERVAL

            st_cur_loss0 = st_total_loss0 / UPDATE_INTERVAL
            st_cur_loss1 = st_total_loss1 / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f} | st_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1, st_cur_loss0, st_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            st_total_loss0 = 0
            st_total_loss1 = 0
            start_time = time.time()
            # attention Maps
            # img_set, _ = \
            #     build_super_images(imgs[-1].cpu(), captions,
            #                        ixtoword, attn_maps, att_sze)
            # if img_set is not None:
            #     im = Image.fromarray(img_set)
            #     fullpath = '%s/attention_maps%d.png' % (image_dir, step)
            #     im.save(fullpath)

        if debug:
            if step == 10:
                break

    return count


def evaluate(dataloader, cnn_model, rnn_model, batch_size, debug=False):

    if not debug:
        cnn_model.eval()

    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    st_total_loss = 0
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
            words_features, sent_code, story_code = cnn_model(st_imgs)
        else:
            batch_size = st_imgs.shape[0]
            words_features = torch.zeros(batch_size*5, 256, 17, 17).cuda()
            sent_code = torch.zeros(batch_size*5, 256).cuda()
            story_code = torch.zeros(batch_size, 256).cuda()
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        word_hidden, sent_hidden = rnn_model.init_hidden(batch_size, cfg.CUDA)
        caption_lens = torch.sum(st_masks, dim=-1).view(-1)
        words_emb, sent_emb, story_emb = rnn_model(st_input_ids.view(-1, st_input_ids.shape[-1]), caption_lens,
                                                   word_hidden, sent_hidden)

        sent_labels = Variable(torch.LongTensor(range(batch_size*video_len)))
        if cfg.CUDA:
            sent_labels = sent_labels.cuda()
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, sent_labels,
                                                 caption_lens, batch_size*video_len)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, sent_labels, batch_size*video_len, cfg.TRAIN.SMOOTH.GAMMA3)
        s_total_loss += (s_loss0 + s_loss1).data

        story_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            story_labels = story_labels.cuda()
        st_loss0, st_loss1 = \
            sent_loss(story_code, story_emb, story_labels, batch_size, cfg.TRAIN.SMOOTH.GAMMA4)
        st_total_loss += (st_loss0 + st_loss1).data

        if debug:
            if step == 50:
                break

    s_cur_loss = s_total_loss/step
    w_cur_loss = w_total_loss/step
    st_cur_loss = st_total_loss/step

    return s_cur_loss, w_cur_loss, st_cur_loss


def build_models(vocab_size, emb_dim, hidden_dim, pretrained_embeddings=None, debug=False):
    # build model ############################################################
    text_encoder = RNN_ENCODER(vocab_size, emb_dim=emb_dim, nhidden=hidden_dim)

    if pretrained_embeddings:
        text_encoder.embeddings.weight.data.copy_(torch.from_numpy(torch.load(pretrained_embeddings)).float())
        text_encoder.embeddings.weight.requires_grad = True

    if debug:
        image_encoder = None
    else:
        image_encoder = CNN_ENCODER(hidden_dim)

    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)

    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        if not debug:
            image_encoder = image_encoder.cuda()

    return text_encoder, image_encoder, start_epoch

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
    output_dir = './output/%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(0)
    cudnn.benchmark = True

    im_input_size = 299
    transform_train = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Resize(im_input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Resize(im_input_size),
        transforms.CenterCrop(im_input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    video_transform_train = functools.partial(video_transform, image_transform=transform_train)
    video_transform_val = functools.partial(video_transform, image_transform=transform_val)

    counter = np.load(os.path.join(dir_path, 'frames_counter.npy'), allow_pickle=True).item()
    print("----------------------------------------------------------------------------------")
    print("Preparing TRAINING dataset")
    base = data.VideoFolderDataset(dir_path, counter=counter, cache=dir_path, min_len=4, mode='train')
    storydataset = data.StoryDataset(base, dir_path, video_transform_train, return_caption=True)

    storydataset.init_mart_vocab()
    print("Built vocabulary of %s words" % len(storydataset.vocab))
    if cfg.MART.vocab_glove_path == '':
        cfg.MART.vocab_glove_path = os.path.join(dir_path, 'martgan_embeddings.mat')
    storydataset.vocab.extract_glove(cfg.MART.raw_glove_path, cfg.MART.vocab_glove_path)
    cfg.MART.pretrained_embeddings = cfg.MART.vocab_glove_path

    storyloader = torch.utils.data.DataLoader(
        storydataset, batch_size=cfg.TRAIN.ST_BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

    base_test = data.VideoFolderDataset(dir_path, counter, dir_path, 4, mode='val')
    testdataset = data.StoryDataset(base_test, dir_path, video_transform_val, return_caption=True)
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=cfg.TRAIN.ST_BATCH_SIZE,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

    n_hidden = 256
    # Train ##############################################################
    text_encoder, image_encoder, start_epoch = build_models(len(storydataset.vocab), cfg.TEXT.DIMENSION, hidden_dim=n_hidden,
                                                            pretrained_embeddings=cfg.MART.pretrained_embeddings, debug=args.debug)
    para = list(text_encoder.parameters())
    if not args.debug:
        for v in image_encoder.parameters():
            if v.requires_grad:
                para.append(v)

    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.DAMSM_LR
        batch_size = cfg.TRAIN.ST_BATCH_SIZE
        eval_batch_size = cfg.TRAIN.ST_BATCH_SIZE

        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(storyloader, image_encoder, text_encoder, batch_size, optimizer, epoch, args.debug)
            print('-' * 89)
            if lr > cfg.TRAIN.DAMSM_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or epoch == cfg.TRAIN.MAX_EPOCH):
                s_loss, w_loss, st_loss = evaluate(testloader, image_encoder, text_encoder, eval_batch_size, args.debug)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, st_loss, lr))
                print('-' * 89)
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')