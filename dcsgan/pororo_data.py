import os, pickle, re, csv
from tqdm import tqdm
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import PIL
from collections import Counter
import nltk
import json


class Vocabulary(object):

    def __init__(self,
                 vocab_threshold,
                 vocab_file,
                 annotations_file,
                 vocab_from_file=False,
                 unk_word="[UNK]",
                 pad_word="[PAD]",
                 start_word="[BOS]",
                 end_word="[EOS]"):
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
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.start_word=start_word
        self.end_word = end_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            print('Reading vocabulary from %s file!' % self.vocab_file)
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab['word2idx']
                self.idx2word = vocab['idx2word']
            print('Vocabulary successfully loaded from %s file!' % self.vocab_file)
        else:
            print("Building voabulary from scratch")
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.unk_word)
        self.add_word(self.pad_word)
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

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, counter = None, cache=None, min_len=4, mode='train', load_images=False, id_file='train_seen_unseen_ids.npy'):
        self.lengths = []
        self.followings = []
        dataset = ImageFolder(folder)
        self.dir_path = folder
        self.total_frames = 0
        self.images = []
        self.labels = np.load(os.path.join(folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        if cache is not None and os.path.exists(cache + 'img_cache' + str(min_len) + '.npy') and os.path.exists(cache + 'following_cache' + str(min_len) +  '.npy'):
            self.images = np.load(cache + 'img_cache' + str(min_len) + '.npy', encoding='latin1')
            self.followings = np.load(cache + 'following_cache' + str(min_len) + '.npy')
        else:
            for idx, (im, _) in enumerate(
                    tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx]
                v_name = img_path.replace(folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > counter[v_name] - min_len:
                    continue
                following_imgs = []
                for i in range(min_len):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(folder, 'img_cache' + str(min_len) + '.npy'), self.images)
            np.save(os.path.join(folder, 'following_cache' + str(min_len) + '.npy'), self.followings)

        # train_id, test_id = np.load(self.dir_path + 'train_test_ids.npy', allow_pickle=True, encoding='latin1')
        train_id, val_id, test_id = np.load(os.path.join(self.dir_path, id_file), allow_pickle=True)
        if mode == 'train':
            orders = train_id
        elif mode =='val':
            orders = val_id[:2320]
        elif mode == 'test':
            orders = test_id
        else:
            raise ValueError

        orders = np.array(orders).astype('int32')
        self.images = self.images[orders]
        self.followings = self.followings[orders]
        print("Total number of clips {}".format(len(self.images)))

        self.image_arrays = {}
        if load_images:
            for idx, (im, _) in enumerate(
                    tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx]
                self.image_arrays[img_path] = im

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0, video_len, 1)[0]
        #print(se*shorter, shorter, (se+1)*shorter)
        return im.crop((0, se * shorter, shorter, (se+1)*shorter)), se

    def __getitem__(self, item):
        lists = [self.images[item]]
        for i in range(len(self.followings[item])):
            lists.append(str(self.followings[item][i]))
        return lists

    def __len__(self):
        return len(self.images)

class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, textvec, transform, return_caption=False, out_dir=None):
        self.dir_path = dataset.dir_path
        self.dataset = dataset
        self.descriptions = np.load(textvec + 'descriptions_vec.npy', allow_pickle=True, encoding='latin1').item()
        self.attributes = np.load(textvec + 'descriptions_attr.npy', allow_pickle=True, encoding='latin1').item()
        self.subtitles = np.load(textvec + 'subtitles_vec.npy', allow_pickle=True, encoding='latin1').item()
        self.descriptions_original = np.load(textvec + 'descriptions.npy', allow_pickle=True, encoding='latin1').item()  
        self.transforms = transform
        self.labels = dataset.labels
        self.return_caption = return_caption
        if self.return_caption:
            self.init_mart_vocab()
            self.max_len = self.tokenize_descriptions()
            print("Max sequence length = %s" % self.max_len)
        else:
            self.vocab = None
        self.out_dir = out_dir


    def tokenize_descriptions(self):
        caption_lengths = []
        self.tokenized_descriptions = {}
        for img_id, descs in self.descriptions_original.items():
            self.tokenized_descriptions[img_id] = [nltk.tokenize.word_tokenize(caption.lower()) for caption in descs]
            caption_lengths.extend([len(c) for c in self.tokenized_descriptions[img_id]])
        return max(caption_lengths) + 2

    def init_mart_vocab(self):

        if os.path.exists(os.path.join(self.dir_path, 'videocap_vocab.pkl')):
            vocab_from_file = True
            vocab_file = os.path.join(self.dir_path, 'videocap_vocab.pkl')
        else:
            vocab_from_file = False
            vocab_file = os.path.join(self.dir_path, 'mart_gan_vocab.pkl')

        self.vocab = Vocabulary(vocab_threshold=5,
                                vocab_file=vocab_file,
                                annotations_file=os.path.join(self.dir_path, 'descriptions.csv'),
                                vocab_from_file=vocab_from_file)

    def save_story(self, output, save_path = './'):
        all_image = []
        images = output['images_numpy']
        texts = output['text']
        for i in range(images.shape[0]):
            all_image.append(np.squeeze(images[i]))
        output = PIL.Image.fromarray(np.concatenate(all_image, axis = 0))
        output.save(save_path + 'image.png')
        fid = open(save_path + 'text.txt', 'w')
        for i in range(len(texts)):
            fid.write(texts[i] +'\n' )
        fid.close()
        return

    def _sentence_to_idx(self, sentence_tokens):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_len
        sentence_tokens = [self.vocab.start_word] + sentence_tokens + [self.vocab.end_word]
        sentence_tokens = sentence_tokens[:max_t_len]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.vocab.pad_word] * (max_t_len - valid_l)
        input_ids = [self.vocab.word2idx.get(t, self.vocab.word2idx[self.vocab.unk_word]) for t in sentence_tokens]

        return input_ids, mask

    def __getitem__(self, item):
        lists = self.dataset[item]
        labels = []
        image = []
        subs = []
        des = []
        attri = []
        text = []
        input_ids = []
        masks= []
        im_sample_idxs = []
        for idx, v in enumerate(lists):
            img_id = str(v).replace('.png','')[2:-1]
            path = self.dir_path + img_id + '.png'
            if self.dataset.image_arrays != {}:
                im = self.dataset.image_arrays[path]
            else:
                if self.out_dir:
                    im = PIL.Image.open(os.path.join(self.out_dir, 'img-%s-%s.png' % (item, idx))).convert('RGB')
                else:
                    im = PIL.Image.open(path)
            if self.out_dir:
                image.append(np.expand_dims(np.array(im), axis = 0))
            else:
                sampled_im, sample_idx = self.dataset.sample_image(im)
                image.append(np.expand_dims(np.array(sampled_im), axis = 0))
                im_sample_idxs.append(sample_idx)
            se = 0
            if len(self.descriptions_original[img_id]) > 1:
                se = np.random.randint(0,len(self.descriptions_original[img_id]),1)
                se = se[0]
            text.append(  self.descriptions_original[img_id][se])
            des.append(np.expand_dims(self.descriptions[img_id][se], axis = 0))
            subs.append(np.expand_dims(self.subtitles[img_id][0], axis = 0))
            labels.append(np.expand_dims(self.labels[img_id], axis = 0))
            attri.append(np.expand_dims(self.attributes[img_id][se].astype('float32'), axis = 0))
            if self.return_caption:
                input_id, mask = self._sentence_to_idx(self.tokenized_descriptions[img_id][se])
                input_ids.append(np.expand_dims(input_id, axis=0))
                masks.append(np.expand_dims(mask, axis=0))

        subs = np.concatenate(subs, axis = 0)
        attri = np.concatenate(attri, axis = 0)
        des = np.concatenate(des, axis = 0)
        labels = np.concatenate(labels, axis = 0)
        image_numpy = np.concatenate(image, axis = 0)
        # image is T x H x W x C
        image = self.transforms(image_numpy)  
        # After transform, image is C x T x H x W
        des = np.concatenate([des, attri], 1)
        ##
        des = torch.tensor(des)
        subs = torch.tensor(subs)
        attri = torch.tensor(attri)
        labels = torch.tensor(labels.astype(np.float32))

        data_item = {'images': image, 'text':text, 'description': des,
                'subtitle': subs, 'images_numpy':image_numpy, 'labels':labels}

        if self.return_caption:
            input_ids = torch.tensor(np.concatenate(input_ids))
            masks = torch.tensor(np.concatenate(masks))
            data_item.update({'input_ids': input_ids, 'masks': masks})

        return data_item

    def __len__(self):
        return len(self.dataset.images)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, textvec, transform, return_caption=False):
        self.dir_path = dataset.dir_path
        self.dataset = dataset
        self.transforms = transform
        self.descriptions = np.load(textvec + 'descriptions_vec.npy', allow_pickle=True, encoding='latin1').item()
        self.attributes =  np.load(textvec + 'descriptions_attr.npy', allow_pickle=True, encoding='latin1').item()
        self.subtitles = np.load(textvec + 'subtitles_vec.npy', allow_pickle=True, encoding='latin1').item()
        self.descriptions_original = np.load(textvec + 'descriptions.npy', allow_pickle=True, encoding='latin1').item()
        self.labels = dataset.labels
        self.return_caption = return_caption
        if self.return_caption:
            self.init_mart_vocab()
            self.max_len = self.tokenize_descriptions()
            print("Max sequence length = %s" % self.max_len)
        else:
            self.vocab = None


    def tokenize_descriptions(self):
        caption_lengths = []
        self.tokenized_descriptions = {}
        for img_id, descs in self.descriptions_original.items():
            self.tokenized_descriptions[img_id] = [nltk.tokenize.word_tokenize(caption.lower()) for caption in descs]
            caption_lengths.extend([len(c) for c in self.tokenized_descriptions[img_id]])
        return max(caption_lengths) + 2

    def _sentence_to_idx(self, sentence_tokens):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_len
        sentence_tokens = sentence_tokens[:max_t_len - 2]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.vocab.pad_word] * (max_t_len - valid_l)
        input_ids = [self.vocab.word2idx.get(t, self.vocab.word2idx[self.vocab.unk_word]) for t in sentence_tokens]

        return input_ids, mask

    def init_mart_vocab(self):

        if os.path.exists(os.path.join(self.dir_path, 'videocap_vocab.pkl')):
            vocab_from_file = True
            vocab_file = os.path.join(self.dir_path, 'videocap_vocab.pkl')
        else:
            vocab_from_file = False
            vocab_file = os.path.join(self.dir_path, 'mart_gan_vocab.pkl')

        self.vocab = Vocabulary(vocab_threshold=5,
                                vocab_file=vocab_file,
                                annotations_file=os.path.join(self.dir_path, 'descriptions.csv'),
                                vocab_from_file=vocab_from_file)

    def __getitem__(self, item):
        path = self.dir_path + str(self.dataset[item][0])[2:-1]
        id = str(self.dataset[item][0]).replace('.png','')[2:-1]
        img_id = id
        im = PIL.Image.open(path)
        image, sample_idx = self.dataset.sample_image(im)
        image = self.transforms(np.array(image))
        subs = self.subtitles[id][0]
        se = 0
        if len(self.descriptions_original[id]) > 1:
            se = np.random.randint(0,len(self.descriptions_original[id]),1)
            se = se[0]
        des = self.descriptions[id][se]
        attri = self.attributes[id][se].astype('float32')
        text = self.descriptions_original[id][se]
        label = self.labels[id].astype(np.float32)
        input_id = None
        mask = None
        if self.return_caption:
            input_id, mask = self._sentence_to_idx(self.tokenized_descriptions[id][se])
            input_id = np.array(input_id)
            mask = np.array(mask)

        lists = self.dataset[item]
        content = []
        attri_content = []
        attri_label = []
        for v in lists:
            id =str(v).replace('.png','')[2:-1]
            se = 0
            if len(self.descriptions[id]) > 1:
                se = np.random.randint(0,len(self.descriptions[id]),1)
                se = se[0]
            content.append(np.expand_dims(self.descriptions[id][se], axis = 0))
            attri_content.append(np.expand_dims(self.attributes[id][se].astype('float32'), axis = 0))
            attri_label.append(np.expand_dims(self.labels[id].astype('float32'), axis = 0))
        content = np.concatenate(content, axis = 0)
        attri_content = np.concatenate(attri_content, axis = 0)
        attri_label = np.concatenate(attri_label, axis = 0)
        content = np.concatenate([content, attri_content, attri_label], 1)
        des = np.concatenate([des, attri])
        ##
        content = torch.tensor(content)

        data_item = {'images': image, 'text':text, 'description': des,
                'subtitle': subs, 'labels':label, 'content': content}

        if self.return_caption:
            input_id = torch.tensor(input_id)
            mask = torch.tensor(mask)
            data_item.update({'input_id': input_id, 'mask':mask})

        return data_item

    def __len__(self):
        return len(self.dataset.images)


class ImageClfDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, im_input_size, mode='train', transform=None, out_img_folder=None):
        self.lengths = []
        self.followings = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        # train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_val_test_ids.npy'), allow_pickle=True)
        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')

        if mode == 'train':
            self.ids = train_ids
            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(im_input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            if mode == 'val':
                self.ids = val_ids[:2304]
            elif mode == 'test':
                self.ids = test_ids
            else:
                raise ValueError

            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(im_input_size),
                    transforms.CenterCrop(im_input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        self.out_dir = out_img_folder

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):
        img_id = self.ids[item]
        img_path = str(self.images[img_id])[2:-1]

        if self.out_dir is not None:
            image = PIL.Image.open(os.path.join(self.out_dir, 'img-%s-0.png' % item)).convert('RGB')
        else:
            image = self.sample_image(PIL.Image.open(os.path.join(self.img_folder, img_path)).convert('RGB'))

        # image = self.sample_image(PIL.Image.open(img_path).convert('RGB'))
        # image = Image.open(os.path.join(self.out_dir, 'img-' + str(item) + '.png')).convert('RGB')
        label = self.labels[img_path.replace('.png', '').replace(self.img_folder + '/', '')]
        return self.transform(image), torch.Tensor(label)

    def __len__(self):
        return len(self.ids)


class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, im_input_size,
                 out_img_folder = '/ssd-playpen/home/adyasha/projects/StoryGAN/pororo_code_mod/output/pororo_both_stageI_r1.0/Test/images-epoch-110/',
                 mode='train',
                 video_len = 5,
                 transform=None):
        self.lengths = []
        self.followings = []
        self.images = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.video_len = video_len

        if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
            self.counter = ''
        else:
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(self.img_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > self.counter[v_name] - (self.video_len-1):
                    continue
                following_imgs = []
                for i in range(self.video_len-1):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(self.img_folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(self.img_folder, 'img_cache4.npy'), self.images)
            np.save(os.path.join(self.img_folder, 'following_cache4.npy'), self.followings)

        # train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_val_test_ids.npy'), allow_pickle=True)
        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)

        if mode == 'train':
            self.ids = train_ids
            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(im_input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            if mode == 'val':
                self.ids = val_ids[:2304]
            elif mode == 'test':
                self.ids = test_ids
            else:
                raise ValueError

            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(im_input_size),
                    transforms.CenterCrop(im_input_size),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        self.out_dir = out_img_folder

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        #print(se*shorter, shorter, (se+1)*shorter)
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):

        img_id = self.ids[item]
        img_paths = [str(self.images[img_id])[2:-1]] + [str(self.followings[img_id][k])[2:-1] for k in range(0, self.video_len-1)]
        if self.out_dir is not None:
            images = [PIL.Image.open(os.path.join(self.out_dir, 'img-%s-%s.png' % (item, k))).convert('RGB') for k in range(self.video_len)]
        else:
            images = [self.sample_image(PIL.Image.open(os.path.join(self.img_folder, path)).convert('RGB')) for path in img_paths]
        labels = [self.labels[path.replace('.png', '').replace(self.img_folder + '/', '')] for path in img_paths]
        # return torch.cat([self.transform(image).unsqueeze(0) for image in images], dim=0), torch.tensor(np.vstack(labels))
        return torch.stack([self.transform(im) for im in images]), torch.tensor(np.vstack(labels))

    def __len__(self):
        return len(self.ids)