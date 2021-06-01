# example of calculating the frechet inception distance
import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from skimage.measure import compare_ssim as ssim
import torch.nn as nn

frame2file = {1: 'disc_dataset_first_frame.json',
              2: 'disc_dataset_second_frame.json',
              3: 'disc_dataset_third_frame.json',
              4: 'disc_dataset_fourth_frame.json',
              5: 'disc_dataset.json'}

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionFeatureExtractor, self).__init__()
        model_ft = models.inception_v3(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.parameters():
            param.requires_grad = False
        self.define_module(model_ft)

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

    def forward(self, x):

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
        # x = x.view(x.size(0), -1)
        # 2048
        return x.squeeze()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(use_pretrained=True, feature_extract=False):
    # model_ft = models.vgg11_bn(pretrained=use_pretrained)
    # input_size = 224
    # model_ft.classifier = model_ft.classifier[:-1]

    model_ft = InceptionFeatureExtractor()
    input_size = 299

    set_parameter_requires_grad(model_ft, feature_extract)
    return model_ft, input_size

def sample_image(im):
    shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
    video_len = int(longer / shorter)
    se = np.random.randint(0, video_len, 1)[0]
    # print(se*shorter, shorter, (se+1)*shorter)
    return im.crop((0, se * shorter, shorter, (se + 1) * shorter))

def eval(args):

    top_1_acc = 0
    top_2_acc = 0

    train_id, val_id, test_id = np.load(os.path.join(args.data_dir, 'train_seen_unseen_ids.npy'), allow_pickle=True)
    ids = val_id if args.mode == 'val' else test_id

    disc_dataset = json.load(open(os.path.join(args.data_dir, frame2file[args.frame])))
    print(len(disc_dataset.keys()), len(val_id), len(test_id))

    img_size = 128
    if args.metric == "cosine":
        model, img_size = initialize_model(feature_extract=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        cosine_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    skipped = 0
    for i, id in tqdm(enumerate(ids[:2304])):
        # query image
        query_img = Image.open(os.path.join(args.output_dir, 'img-%s-0.png' % i)).resize((img_size, img_size, ))
        query_pix = np.array(query_img.getdata()).reshape(query_img.size[0], query_img.size[1], 3)
        # load choices
        try:
            choices = [disc_dataset[str(id)]["target"]] + disc_dataset[str(id)]["choices"]
        except KeyError:
            skipped += 1
            continue
        # print(choices)
        choice_imgs = [sample_image(Image.open(os.path.join(args.data_dir, c))).resize((img_size, img_size, )) for c in choices]
        choice_pix = [np.array(c.getdata()).reshape(c.size[0], c.size[1], 3) for c in choice_imgs]
        # compute ssims
        # print(query_pix[:, 0, 0], choice_pix[0][:, 0, 0])
        if args.metric == "ssim":
            ssims = [ssim(query_pix, im, data_range=im.max() - im.min(), multichannel=True) for im in choice_pix]
            ranks = [i for i, score in sorted(enumerate(ssims), key=lambda t: t[1], reverse=True)]
        elif args.metric == "cosine":
            input_tensor = torch.cat([preprocess(img).unsqueeze(0) for img in [query_img] + choice_imgs], dim=0)
            features = model(input_tensor.to(device))
            sims = cosine_fn(features[0, :].repeat(5, 1), features[1:, :]).cpu().numpy()
            ranks = [i for i, score in sorted(enumerate(sims), key=lambda t: t[1], reverse=True)]
        else:
            raise ValueError

        if 0 in ranks[0:1]:
            top_1_acc += 1
        if 0 in ranks[0:2]:
            top_2_acc += 1

        if i%100 == 0:
            print(float(top_1_acc)/(i+1), float(top_2_acc)/(i+1))

    print(top_1_acc, top_2_acc)
    print(skipped)

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Evaluate Discriminative Dataset')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--mode', default='val')
    parser.add_argument('--frame', type=int, default=1)
    args = parser.parse_args()

    eval(args)

