from __future__ import print_function
from __future__ import division
import torch.nn as nn
from torchvision import models
import torch
from classifier.dataloader import ImageDataset, StoryImageDataset
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy
import os
import PIL
import torchvision.utils as vutils
import argparse
from sklearn.metrics import classification_report, accuracy_score

epsilon = 1e-7

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet50
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
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

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

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
    for i in tqdm(range(x.shape[0])):
        frames = x[i, :, :, :, :]
        frames = np.swapaxes(frames, 0, 1)
        # frames = torch.Tensor(frames).view(-1, 3, 64, 64)
        # frames = torch.nn.functional.upsample(frames, size=(img_size, img_size), mode="bilinear")

        # vutils.save_image(vutils.make_grid(torch.Tensor(frames).view(-1, 3, 64, 64), 1, padding=0), 'sequence-2.png')
        all_images = images_to_numpy(vutils.make_grid(torch.Tensor(frames).view(-1, 3, 64, 64), 1, padding=0))
        # all_images = images_to_numpy(vutils.make_grid(frames, 1, padding=0))
        # print(all_images.shape)
        for j, idx in enumerate(range(64, all_images.shape[0] + 1, 64)):
            output = PIL.Image.fromarray(all_images[idx-64: idx, :, :])
            output.save(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))
            img = PIL.Image.open(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))
            if img_size != 64:
                img = img.resize((img_size, img_size,))
            img.save(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))

def evaluate_gt(root_image_dir, model_name, model_path):

    # Number of classes in the dataset
    num_classes = 9
    #   when True we only update the reshaped layer params
    feature_extract = False
    video_len = 5
    n_channels = 3

    running_corrects = 0
    running_recalls = 0
    total_positives = 0

    phase = 'eval'
    is_inception = True if model_name == 'inception' else False
    data_dir = "../../pororo_png"

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    model_ft.load_state_dict(torch.load(model_path))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluate mode
    image_dataset = ImageDataset(data_dir, input_size, mode='val')
    print("Number of samples in evaluation set: %s" % len(image_dataset))
    batch_size = 32

    # Create validation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Number of batches in evaluation dataloader: %s" % len(dataloader))

    all_predictions = []
    all_labels = []
    story_accuracy = 0
    image_accuracy = 0

    # Iterate over data.
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(phase == 'train'):

            outputs = model_ft(inputs)
            if model_name == 'imgD':
                outputs = model_ft.cate_classify(outputs).squeeze()
            preds = torch.round(nn.functional.sigmoid(outputs))
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # statistics
        iter_corrects = torch.sum(preds == labels.float().data)
        xidxs, yidxs = torch.where(labels.data == 1)
        # print(xidxs, yidxs)
        # print([labels.data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)])
        iter_recalls = sum(
            [x.item() for x in
             [labels.float().data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)]])
        total_positives += xidxs.size(0)

        for l, p in zip(labels, preds):
            if torch.all(torch.eq(l.float().data, p)):
                image_accuracy += 1

        running_corrects += iter_corrects
        running_recalls += iter_recalls

    epoch_acc = running_corrects * 100 / (len(image_dataset) * num_classes)
    epoch_recall = running_recalls * 100 / total_positives
    print('{} Acc: {:.4f} Recall: {:.4f}%'.format(phase, epoch_acc, epoch_recall))
    print('{} Story Exact Match Acc: {:.4f}%'.format(phase, float(story_accuracy) * 100 / len(image_dataset)))
    print('{} Image Exact Match Acc: {:.4f}%'.format(phase, float(image_accuracy) * 100 / len(image_dataset)))

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_predictions.shape, all_labels.shape, image_accuracy, len(image_dataset))
    preds = np.round(1 / (1 + np.exp(-all_predictions)))
    print(classification_report(all_labels, preds, digits=4))

    for i in range(0, 9):
        print("Character %s" % i)
        print(classification_report(all_labels[:, i], preds[:, i]))

    # Inception Score
    all_predictions = all_predictions + epsilon
    py = np.mean(all_predictions, axis=0)
    print(py, py.shape)
    split_scores = []
    splits = 10
    N = all_predictions.shape[0]
    for k in range(splits):
        part = all_predictions[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []

        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    print("InceptionScore", np.mean(split_scores), np.std(split_scores))


def evaluate(root_image_dir, epoch_start, epoch_end, model_name, model_path):

    # Number of classes in the dataset
    num_classes = 9
    #   when True we only update the reshaped layer params
    feature_extract = False
    video_len = 5
    n_channels = 3

    phase = 'eval'
    is_inception = True if model_name == 'inception' else False
    data_dir = "../../pororo_png"

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    model_ft.load_state_dict(torch.load(model_path))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluate mode

    for epoch in range(epoch_start, epoch_end+1, 10):

        print("Epoch: %s" % epoch)
        if not os.path.exists(os.path.join(root_image_dir, 'images-epoch-%s/' % epoch)):
            numpy_to_img(os.path.join(root_image_dir, 'images-epoch-%s.npy' % epoch),
                         os.path.join(root_image_dir,'images-epoch-%s/' % epoch), input_size)

        # Create training and validation datasets
        image_dataset = StoryImageDataset(data_dir, input_size,
                                          out_img_folder=os.path.join(root_image_dir, 'images-epoch-%s/' % epoch),
                                          mode='val')
        print("Number of samples in evaluation set: %s" % len(image_dataset))
        batch_size = 8

        # Create validation dataloaders
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print("Number of batches in evaluation dataloader: %s" % len(dataloader))

        all_predictions = []
        all_labels = []
        all_labels = []
        story_accuracy = 0
        image_accuracy = 0

        running_corrects = 0
        running_recalls = 0
        total_positives = 0

        # Iterate over data.
        for i, (inputs, labels) in tqdm(enumerate(dataloader)):

            inputs = inputs.view(batch_size * video_len, n_channels, inputs.shape[-2], inputs.shape[-1])
            labels = labels.view(batch_size * video_len, labels.shape[-1])
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                outputs = model_ft(inputs)
                if model_name == 'imgD':
                    outputs = model_ft.cate_classify(outputs).squeeze()
                preds = torch.round(nn.functional.sigmoid(outputs))
                all_predictions.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            # statistics
            iter_corrects = torch.sum(preds == labels.float().data)
            xidxs, yidxs = torch.where(labels.data == 1)
            # print(xidxs, yidxs)
            # print([labels.data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)])
            iter_recalls = sum(
                [x.item() for x in [labels.float().data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)]])
            total_positives += xidxs.size(0)

            labels = labels.view(batch_size, video_len, labels.shape[-1])
            preds = preds.view(batch_size, video_len, labels.shape[-1])

            for label, pred in zip(labels, preds):
                if torch.all(torch.eq(label.float().data, pred)):
                    story_accuracy += 1
                for l, p in zip(label, pred):
                    if torch.all(torch.eq(l.float().data, p)):
                        image_accuracy += 1

            running_corrects += iter_corrects
            running_recalls += iter_recalls

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        print(all_predictions.shape, all_labels.shape, image_accuracy, len(image_dataset))
        preds = np.round(1 / (1 + np.exp(-all_predictions)))
        print(classification_report(all_labels, all_predictions, digits=4))
        print("Accuracy: ", accuracy_score(all_labels, preds))

        epoch_acc = float(running_corrects) * 100 / (all_labels.shape[0] * all_labels.shape[1])
        epoch_recall = float(running_recalls) * 100 / total_positives
        print('Manually calculated accuracy: ', epoch_acc)
        print('{} Acc: {:.4f} Recall: {:.4f}%'.format(phase, accuracy_score(all_labels, preds), epoch_recall))
        print('{} Story Exact Match Acc: {:.4f}%'.format(phase, story_accuracy*100/len(image_dataset)))
        print('{} Image Exact Match Acc: {:.4f}%'.format(phase, image_accuracy * 100 / (len(image_dataset)*video_len)))

        np.save(os.path.join(root_image_dir, 'epoch-%s-prediction-probs.npy' % epoch), 1 / (1 + np.exp(-all_predictions)))
        np.save(os.path.join(root_image_dir, 'labels.npy'), all_labels)

        # Inception Score
        all_predictions = all_predictions + epsilon
        py = np.mean(all_predictions, axis=0)
        print(py, py.shape)
        split_scores = []
        splits = 10
        N= all_predictions.shape[0]
        for k in range(splits):
            part = all_predictions[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []

            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        print("InceptionScore", np.mean(split_scores), np.std(split_scores))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate for Character Recall & InceptionScore')
    parser.add_argument('--image_dir',  type=str, required=True)
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--epoch_end', type=int, default=120)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--ground_truth', action='store_true')
    args = parser.parse_args()

    if args.ground_truth:
        evaluate_gt(args.image_dir, args.model_name, args.model_path)
    else:
        evaluate(args.image_dir, args.epoch_start, args.epoch_end, args.model_name, args.model_path)

    # numpy_to_img(os.path.join(args.image_dir, 'images-epoch-%s.npy' % args.epoch_start),
    #              os.path.join(args.image_dir, 'images-epoch-%s/' % args.epoch_start), 299)

