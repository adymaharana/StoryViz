import os, re
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, im_input_size, mode='train'):
        self.lengths = []
        self.followings = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        # train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_val_test_ids.npy'), allow_pickle=True)
        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        if mode == 'train':
            self.ids = train_ids
            self.transform = transforms.Compose([
                transforms.Resize(im_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.ids = val_ids
            self.transform = transforms.Compose([
                transforms.Resize(im_input_size),
                transforms.CenterCrop(im_input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        #self.out_dir = '/ssd-playpen/home/adyasha/projects/StoryGAN/pororo_code_mod/output/pororo_both_stageI_r1.0/Test/epoch_110'
        # self.out_dir = '/ssd-playpen/home/adyasha/projects/StoryGAN/code_mart/output/pororo_mart_stageI_r1.0/Test/epoch-90'

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        #print(se*shorter, shorter, (se+1)*shorter)
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):

        img_id = self.ids[item]
        img_path = self.img_dataset.imgs[img_id][0]
        image = self.sample_image(Image.open(img_path).convert('RGB'))
        # image = Image.open(os.path.join(self.out_dir, 'img-' + str(item) + '.png')).convert('RGB')
        label = self.labels[img_path.replace('.png', '').replace(self.img_folder + '/', '')]
        return self.transform(image), torch.Tensor(label)

    def __len__(self):
        return len(self.ids)


class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, im_input_size,
                 out_img_folder = '/ssd-playpen/home/adyasha/projects/StoryGAN/pororo_code_mod/output/pororo_both_stageI_r1.0/Test/images-epoch-110/',
                 mode='train',
                 video_len = 5):
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

        #train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_val_test_ids.npy'), allow_pickle=True)
        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)

        if mode == 'train':
            self.ids = train_ids
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
            images = [Image.open(os.path.join(self.out_dir, 'img-%s-%s.png' % (item, k))).convert('RGB') for k in range(self.video_len)]
        else:
            images = [self.sample_image(Image.open(os.path.join(self.img_folder, path)).convert('RGB')) for path in img_paths]
        labels = [self.labels[path.replace('.png', '').replace(self.img_folder + '/', '')] for path in img_paths]
        return torch.cat([self.transform(image).squeeze(0) for image in images], dim=0), torch.tensor(np.vstack(labels))

    def __len__(self):
        return len(self.ids)
