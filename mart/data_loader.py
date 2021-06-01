import nltk
import os, csv
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import re, copy
import pickle
import os.path
from collections import Counter
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
import torchvision.utils as vutils

def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1, 2, 0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def numpy_to_img(numpy_file, outdir, img_size):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    x = np.load(numpy_file)
    print("Numpy image file shape: ", x.shape)
    for i in range(x.shape[0]):
        frames = x[i, :, :, :, :]
        frames = np.swapaxes(frames, 0, 1)

        # vutils.save_image(vutils.make_grid(torch.Tensor(frames).view(-1, 3, 64, 64), 1, padding=0), 'sequence-2.png')
        all_images = images_to_numpy(vutils.make_grid(torch.Tensor(frames).view(-1, 3, 64, 64), 1, padding=0))
        # print(all_images.shape)
        for j, idx in enumerate(range(64, all_images.shape[0] + 1, 64)):
            output = Image.fromarray(all_images[idx-64: idx, :, :])
            output.save(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))
            img = Image.open(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))
            if img_size != 64:
                img = img.resize((img_size, img_size,))
            img.save(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))

def prepare_batch_inputs(batch, bsz, device, non_blocking=False):
    batch_inputs = dict()
    for k, v in batch.items():
        if k =='input_tokens' or k == 'caption':
            continue
        assert bsz == len(v), (bsz, k, v.shape)
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        else:  # all non-tensor values
            batch_inputs[k] = v
    return batch_inputs

def caption_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    HOW to batch clip-sentence pair?
    1) directly copy the last sentence, but do not count them in when back-prop OR
    2) put all -1 to their text token label, treat
    """
    # Step2: batching each steps individually in the batches
    collated_step_batch = []
    for step_idx in range(len(batch)):
        collated_step = step_collate([e[step_idx] for e in batch])
        collated_step_batch.append(collated_step)
    return collated_step_batch

def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch

class MyInceptionFeatureExtractor(nn.Module):
    def __init__(self, model, transform_input=False):
        super(MyInceptionFeatureExtractor, self).__init__()

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
        # stop where you want, copy paste from the model def
        self.transform_input = transform_input

    def forward(self, x):
        # --> fixed-size input: batch x 3 x 299 x 299
        if self.transform_input:
            x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
            x = x.clone()
            x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

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
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        return x


def get_loader(transform,
               data_dir,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='../out/vocab.pkl',
               vocab_from_file=True,
               num_workers=0,
               pred_img_dir=None):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary.
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """

    assert mode in ['train', 'val', 'test'], "mode must be one of 'train', 'val' or 'test'."
    if vocab_from_file == False: assert mode == 'train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    # if mode == 'train':
    #     if vocab_from_file == True: assert os.path.exists(
    #         vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."

    if mode in ['test', 'val']:
        # assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."

    annotations_file = os.path.join(data_dir, "descriptions.csv")
    frame_counter = np.load(os.path.join(data_dir, 'frames_counter.npy'), allow_pickle=True).item()


    # COCO caption dataset.
    dataset = CaptioningDataset(transform=transform,
                                mode=mode,
                                batch_size=batch_size,
                                vocab_threshold=vocab_threshold,
                                vocab_file=vocab_file,
                                annotations_file=annotations_file,
                                vocab_from_file=vocab_from_file,
                                img_folder=data_dir,
                                frame_counter=frame_counter,
                                cache_dir=data_dir,
                                load_images=False,
                                pred_img_dir=pred_img_dir)

    if mode == 'train':

        train_sampler = data.sampler.RandomSampler(dataset)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset,
                                      num_workers=num_workers,
                                      sampler=train_sampler,
                                      batch_size=batch_size,
                                      drop_last=True)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      drop_last=True)

    return data_loader

class Vocabulary(object):

    def __init__(self,
                 vocab_threshold,
                 max_t_len,
                 vocab_file='../../StoryGAN/pororo_png/vocab.pkl',
                 annotations_file='../../StoryGAN/pororo_png/descriptions.csv',
                 vocab_from_file=False,
                 start_word="[BOS]",
                 end_word="[EOS]",
                 unk_word="[UNK]",
                 pad_word="[PAD]",
                 cls_word="[CLS]",
                 sep_word="[SEP]",
                 vid_word="[VID]",
                 ):
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.cls_word = cls_word
        self.sep_word = sep_word
        self.vid_word = vid_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.max_t_len = max_t_len
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab['word2idx']
                self.idx2word = vocab['idx2word']
            print('Vocabulary successfully loaded from %s file!' % self.vocab_file)
        else:
            print("Building vocabulary from scratch")
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_word(self.pad_word)
        self.add_word(self.cls_word)
        self.add_word(self.sep_word)
        self.add_word(self.vid_word)
        self.add_captions()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        counter = Counter()
        with open(self.annotations_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            print("Tokenizing captions")
            for i, row in tqdm(enumerate(csv_reader)):
                _, _, caption = row
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                counter.update(tokens)

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def load_glove(self, filename):
        """ returns { word (str) : vector_embedding (torch.FloatTensor) }
        """
        glove = {}
        with open(filename) as f:
            for line in tqdm(f.readlines()):
                values = line.strip("\n").split(" ")  # space separator
                word = values[0]
                vector = np.asarray([float(e) for e in values[1:]])
                glove[word] = vector
        return glove

    def extract_glove(self, raw_glove_path, vocab_glove_path, glove_dim=300):

        if os.path.exists(vocab_glove_path):
            print("Pre-extracted embedding matrix exists at %s" % vocab_glove_path)
        else:
            # Make glove embedding.
            print("Loading glove embedding at path : {}.\n".format(raw_glove_path))
            glove_full = self.load_glove(raw_glove_path)
            print("Glove Loaded, building word2idx, idx2word mapping.\n")
            idx2word = {v: k for k, v in self.word2idx.items()}

            glove_matrix = np.zeros([len(self.word2idx), glove_dim])
            glove_keys = glove_full.keys()
            for i in tqdm(range(len(idx2word))):
                w = idx2word[i]
                w_embed = glove_full[w] if w in glove_keys else np.random.randn(glove_dim) * 0.4
                glove_matrix[i, :] = w_embed
            print("vocab embedding size is :", glove_matrix.shape)
            torch.save(glove_matrix, vocab_glove_path)

    def _tokenize_pad_sentence(self, sentence):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_t_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [self.start_word] + sentence_tokens + [self.end_word]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.pad_word] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class CaptioningDataset(data.Dataset):

    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, annotations_file,
                 vocab_from_file, img_folder, frame_counter,
                 max_v_len = 64 + 2, cache_dir=None, min_len=4, load_images=False, pred_img_dir=None):

        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size

        self.img_folder = img_folder
        self.img_dataset = ImageFolder(img_folder)
        self.video_len = min_len + 1

        self.images = []
        self.followings = []
        if cache_dir is not None and \
                os.path.exists(os.path.join(cache_dir, 'img_cache' + str(min_len) + '.npy')) and \
                os.path.exists(os.path.join(cache_dir, 'following_cache' + str(min_len) +  '.npy')):
            self.images = np.load(os.path.join(cache_dir, 'img_cache' + str(min_len) + '.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(cache_dir, 'following_cache' + str(min_len) + '.npy'))
        else:
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(img_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > frame_counter[v_name] - min_len:
                    continue
                following_imgs = []
                for i in range(min_len):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(img_folder, ''))
                self.followings.append(following_imgs)
            np.save(img_folder + 'img_cache' + str(min_len) + '.npy', self.images)
            np.save(img_folder + 'following_cache' + str(min_len) + '.npy', self.followings)


        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.max_v_len = max_v_len
        self.IGNORE = -1

        self.annotations = {}
        with open(annotations_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                episode_name, frame_id, caption = row
                self.annotations[os.path.join(episode_name, frame_id + '.png')] = caption

        print('Obtaining caption lengths...')
        all_tokens = [nltk.tokenize.word_tokenize(str(
            self.annotations[os.path.join(self.img_dataset.imgs[idx][0].split('/')[-2],
                                          self.img_dataset.imgs[idx][0].split('/')[-1])]
        ).lower()) for idx in tqdm(train_ids)]
        self.caption_lengths = [len(token) for token in all_tokens]
        self.max_t_len = max(self.caption_lengths) + 2
        print("Maximum caption length {} and Maximum video length {}".format(self.max_t_len, self.max_v_len))

        if self.mode == 'train':
            self.ids = self.train_ids
        elif self.mode =='val':
            self.ids = self.val_ids
        elif self.mode == 'test':
            self.ids = self.test_ids
        print("Total number of clips {}".format(len(self.ids)))

        self.vocab = Vocabulary(vocab_threshold, self.max_t_len, vocab_file, annotations_file, vocab_from_file)
        self.vocab_size = len(self.vocab.word2idx)

        if load_images:
            self.image_arrays = {}
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Loading all images")):
                img_path, _ = self.img_dataset.imgs[idx]
                self.image_arrays[img_path] = Image.open(os.path.join(self.img_folder, img_path)).convert('RGB')
        self.load_images = load_images

        # self.out_dir = '/ssd-playpen/home/adyasha/projects/StoryGAN/src/output/pororo_transformer_stageI_r1.0/Test/images-epoch-110/'
        # self.out_dir = None

        if pred_img_dir:
            assert mode in ['val', 'test']
            self.pred_img_dir = pred_img_dir
            if not os.path.exists(pred_img_dir):
                root_img_dir, sub_dir = os.path.split(pred_img_dir)
                img_np_file = os.path.join(root_img_dir, sub_dir + '.npy')
                if os.path.exists(img_np_file):
                    numpy_to_img(img_np_file, pred_img_dir, 299)
        else:
            self.pred_img_dir = None

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        #print(se*shorter, shorter, (se+1)*shorter)
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def frame_caption_to_feature(self, raw_caption):
        text_tokens, text_mask = self.vocab._tokenize_pad_sentence(str(raw_caption).lower())
        # print('Length of text tokens %s' % len(text_tokens))
        video_tokens = [self.vocab.cls_word] + [self.vocab.vid_word] * (self.max_v_len-2) + [self.vocab.sep_word]
        video_mask = [1] * self.max_v_len
        input_tokens = video_tokens + text_tokens
        input_ids = [self.vocab.word2idx.get(t, self.vocab.word2idx[self.vocab.unk_word]) for t in input_tokens]
        input_labels = \
            [self.IGNORE] * len(video_tokens) + \
            [self.IGNORE if m == 0 else tid for tid, m in zip(input_ids[-len(text_mask):], text_mask)][1:] + \
            [self.IGNORE]
        input_mask = video_mask + text_mask
        token_type_ids = [0] * self.max_v_len + [1] * self.max_t_len
        data = dict(
            input_tokens=input_tokens,
            # model inputs
            input_ids=np.array(input_ids).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64),
            input_mask=np.array(input_mask).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
        )
        return data

    def __getitem__(self, index):
        # obtain image and caption if in training mode
        # if self.mode == 'train':
        data_id = self.ids[index]
        image_seq_paths = [self.images[data_id].decode('utf-8')]
        for img_file in self.followings[data_id]:
            image_seq_paths.append(img_file.decode('utf-8'))
        ann_ids = [os.path.join(img_path.split('/')[-2], img_path.split('/')[-1]) for img_path in image_seq_paths]
        raw_captions = [self.annotations[ann_id] for ann_id in ann_ids]

        # Convert image to tensor and pre-process using transform
        if self.load_images:
            images = [self.image_arrays[img_path] for img_path in image_seq_paths]
        else:
            if self.pred_img_dir:
                images = [Image.open(os.path.join(self.pred_img_dir, 'img-%s-%s.png' % (index, k))).convert('RGB') for k in
                          range(self.video_len)]
            else:
                images = [self.sample_image(Image.open(os.path.join(self.img_folder, img_path)).convert('RGB'))
                      for img_path in image_seq_paths]
        # images = torch.stack([self.transform(image) for image in images])

        # Convert caption to tensor of word ids.
        story_features = []
        for i, raw_caption in enumerate(raw_captions):
            data = self.frame_caption_to_feature(raw_caption)
            data["image"] = self.transform(images[i])
            data["caption"] = raw_caption
            story_features.append(data)

        # print([c.shape for c in captions])
        # return pre-processed image and caption tensors
        return story_features

        # # obtain image if in test mode
        # else:
        #     data_id = self.val_ids[index]
        #     image_seq_paths = [self.images[data_id].decode('utf-8')]
        #     for i in range(len(self.followings[data_id])):
        #         image_seq_paths.append(self.followings[data_id][i].decode('utf-8'))
        #     # print(image_seq_paths)
        #     ann_ids = [os.path.join(img_path.split('/')[-2], img_path.split('/')[-1]) for img_path in image_seq_paths]
        #     raw_captions = [self.annotations[ann_id] for ann_id in ann_ids]
        #
        #     # Convert image to tensor and pre-process using transform
        #     orig_images = [self.sample_image(Image.open(os.path.join(self.img_folder, img_path)).convert('RGB'))
        #                    for img_path in image_seq_paths]
        #     images = [self.transform(image) for image in orig_images]
        #
        #     # return original image and pre-processed image tensor
        #     return [np.array(img) for img in orig_images], torch.stack(images), raw_captions

    def __len__(self):
        return len(self.ids)
