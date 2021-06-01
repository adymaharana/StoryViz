import torch
from torchvision.datasets import ImageFolder
import numpy as np
import os, re
from tqdm import tqdm
import PIL
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import transforms
import pickle

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

class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, cache=None, min_len=4, transform=None, n_samples=5):
        self.lengths = []
        self.followings = []
        dataset = ImageFolder(folder)
        self.dir_path = folder
        self.total_frames = 0
        self.images = []
        if cache is not None and os.path.exists(os.path.join(cache, 'img_cache' + str(min_len) + '.npy')):
            self.images = np.load(os.path.join(cache, 'img_cache' + str(min_len) + '.npy'), encoding='latin1')
        else:
            for idx, (im, _) in enumerate(
                    tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx]
                self.images.append(img_path.replace(folder, ''))

        if not os.path.exists(os.path.join(folder, 'path2idx.pkl')):
            path2idx = {}
            for idx, img_path in tqdm(enumerate(self.images)):
                path2idx[img_path] = idx
            with open(os.path.join(folder, 'path2idx.pkl'), 'wb') as f:
                pickle.dump(path2idx, f)

        print("Total number of images {}".format(len(self.images)))

        if transform is not None:
            self.transform = transform
        self.n_samples = n_samples

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        #print(se*shorter, shorter, (se+1)*shorter)
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):
        v = self.images[item]
        img_id = str(v).replace('.png', '')[2:-1]
        path = self.dir_path + img_id + '.png'
        im = PIL.Image.open(path).convert('RGB')
        # return torch.stack([torch.tensor(np.array(self.sample_image(im))) for _ in range(5)])
        if self.transform:
            return torch.stack([self.transform(self.sample_image(im)) for _ in range(self.n_samples)])
        else:
            return torch.stack([torch.tensor(np.array(self.sample_image(im))) for _ in range(self.n_samples)])

    def __len__(self):
        return len(self.images)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    if model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model selection")
        exit()

    return model_ft, input_size

def main(args):

    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_dataset = VideoFolderDataset(args.data_dir, args.data_dir, transform=train_transform)
    print("Number of samples in evaluation set: %s" % len(image_dataset))
    batch_size = 4
    n_samples = 5

    print(image_dataset[0].shape, image_dataset[0].type())
    print(image_dataset[0])

    # Create validation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    print("Number of batches in dataloader: %s" % len(dataloader))

    num_classes = 9
    model_ft, input_size = initialize_model(args.model_name, num_classes, feature_extract=True, use_pretrained=False)
    model_ft.load_state_dict(torch.load(args.model_path))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluate mode

    model = MyInceptionFeatureExtractor(model_ft)

    all_features = []
    for images in tqdm(dataloader):
        with torch.set_grad_enabled(False):
            # images = images.view(batch_size*video_len, images.shape[-3], images.shape[-2], images.shape[-1])
            print(images.shape)
            bsz = images.shape[0]
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.
            outputs = model(images.view(n_samples*bsz, images.shape[-3], images.shape[-2], images.shape[-1]).to(device))
            print(outputs.shape)
            all_features.append(outputs.view(bsz, n_samples, images.shape[-3], images.shape[-2], images.shape[-1]).detach().cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    print(all_features.shape)
    np.save(os.path.join(args.data_dir, 'img_features_' + args.model_name + '.npy'), all_features)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate for Character Recall & InceptionScore')
    parser.add_argument('--data_dir',  type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()


    main(args)