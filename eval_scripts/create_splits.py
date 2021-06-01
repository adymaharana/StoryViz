import numpy as np
from torchvision.datasets import ImageFolder
import os, re
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_poroqa_character_freq(id_file, label_file, img_folder):

    dataset = ImageFolder(img_folder)
    train_ids, val_ids, test_ids = np.load(id_file, allow_pickle=True)
    labels = np.load(label_file, allow_pickle=True, encoding='latin1').item()

    all_labels = []
    for idx in train_ids:
        img_path, _ = dataset.imgs[idx]
        img_id = img_path.replace(img_folder + '/', '').replace('.png', '')
        label = labels[img_id]
        all_labels.append(label)

    total = np.sum(np.array(all_labels), axis=0)
    train_normalized = total / np.sum(total)

    for k, ids in enumerate((val_ids, test_ids)):
        all_labels = []
        for idx in ids:
            img_path, _ = dataset.imgs[idx]
            img_id = img_path.replace(img_folder + '/', '').replace('.png','')
            label = labels[img_id]
            all_labels.append(label)
        total = np.sum(np.array(all_labels), axis = 0)
        normalized = total/np.sum(total)
        diffs = [abs(t-k) for t, k in zip(train_normalized, normalized)]
        if max(diffs) > 0.05:
            return False

    characters = ['Pororo', 'Loopy', 'Crong', 'Eddy', 'Poby', 'Petty', 'Tongtong', 'Rody', 'Harry']
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors = ['g', 'b', 'r']
    legend_labels = ['Train', 'Validation', 'Test']
    x = np.array(list(range(0, len(characters))))
    offsets = [-0.2, 0, +0.2]
    ax = plt.subplot(1, 1, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(characters)
    for k, ids in enumerate((train_ids, val_ids, test_ids)):
        all_labels = []
        for idx in ids:
            img_path, _ = dataset.imgs[idx]
            img_id = img_path.replace(img_folder + '/', '').replace('.png','')
            label = labels[img_id]
            all_labels.append(label)

        total = np.sum(np.array(all_labels), axis = 0)
        print(total/np.sum(total))
        ax.bar(x+offsets[k], total/np.sum(total), width = 0.25, label=legend_labels[k], color=colors[k])
        # sns.barplot(x+offsets[k], total/max(total))

    # plt.xlabel('PororoSV Characters')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.savefig('pororoqa_character_frequency.png', dpi=300, bbox_inches='tight')

    return True


folder = '../pororo_png'
dataset = ImageFolder(folder)
counter = np.load(os.path.join(folder, 'frames_counter.npy'), allow_pickle=True).item()
dirs = list(set([os.path.split(dataset.imgs[idx][0])[0] for idx in range(len(dataset))]))
print("Found %s directories/episodes" % len(dirs))

min_len = 4


def create_unseen_splits():
    # Iterate until difference between normalized frequencies is less than 5% of target dataset.
    dir_idxs = list(range(len(dirs)))

    for _ in tqdm(range(50)):

        train_idxs = random.sample(dir_idxs, k=int(len(dirs)*0.7))
        val_idxs = random.sample([idx for idx in dir_idxs if idx not in train_idxs], k=int(len(dirs)*0.15))
        test_idxs = [idx for idx in dir_idxs if idx not in train_idxs and idx not in val_idxs]

        train_dirs = [dirs[idx] for idx in train_idxs]
        val_dirs = [dirs[idx] for idx in val_idxs]
        test_dirs = [dirs[idx] for idx in test_idxs]

        images = np.load(os.path.join(folder, 'img_cache' + str(min_len) + '.npy'), encoding='latin1')

        train_seq_idxs = []
        val_seq_idxs = []
        test_seq_idxs = []

        for i, img_path in enumerate(images):
            dir_name = os.path.join(folder, str(os.path.split(img_path)[0])[2:-1])
            if dir_name in train_dirs:
                train_seq_idxs.append((i))
            elif dir_name in val_dirs:
                val_seq_idxs.append(i)
            elif dir_name in test_dirs:
                test_seq_idxs.append(i)
            else:
                print(dir_name)
        np.save(os.path.join(folder, 'train_val_test_ids.npy'), (train_seq_idxs, val_seq_idxs, test_seq_idxs))
        okay = plot_poroqa_character_freq(os.path.join(folder, 'train_val_test_ids.npy'), os.path.join(folder, 'labels.npy'), folder)
        if okay:
            print("%s, %s and %s sequences in training, validation and test sets." % (len(train_seq_idxs), len(val_seq_idxs), len(test_seq_idxs)))
            break


def create_splits_v1():
    # Iterate until difference between normalized frequencies is less than 5% of target dataset.

    train_v0_ids, seen_ids = np.load(os.path.join(folder, 'train_test_ids.npy'), allow_pickle=True, encoding='latin1')
    train_v0_dirs = list(set([os.path.split(dataset.imgs[idx][0])[0] for idx in train_v0_ids]))
    seen_dirs = list(set([os.path.split(dataset.imgs[idx][0])[0] for idx in seen_ids]))
    print(len(train_v0_dirs), len(set(train_v0_dirs).intersection(set(seen_dirs))))
    train_v0_dir_idxs = list(range(len(train_v0_dirs)))

    for _ in tqdm(range(50)):

        train_v1_dir_idxs = random.sample(train_v0_dir_idxs, k=int(len(dirs) * 0.8))
        unseen_dir_idxs = [idx for idx in train_v0_dir_idxs if idx not in train_v1_dir_idxs]

        train_v1_dirs = [train_v0_dirs[idx] for idx in train_v1_dir_idxs]
        unseen_dirs = [train_v0_dirs[idx] for idx in unseen_dir_idxs]
        print('%s and %s directories for training and unseen test (test2)' % (len(train_v1_dir_idxs), len(unseen_dirs)))

        images = np.load(os.path.join(folder, 'img_cache' + str(min_len) + '.npy'), encoding='latin1')

        train_v1_seq_idxs = []
        unseen_seq_idxs = []

        for i, img_path in enumerate(images):
            if i in seen_ids:
                continue
            dir_name = os.path.join(folder, str(os.path.split(img_path)[0])[2:-1])
            if dir_name in train_v1_dirs:
                train_v1_seq_idxs.append(i)
            elif dir_name in unseen_dirs:
                unseen_seq_idxs.append(i)
            else:
                pass
        np.save(os.path.join(folder, 'train_seen_unseen_ids.npy'), (train_v1_seq_idxs, seen_ids, unseen_seq_idxs))
        okay = plot_poroqa_character_freq(os.path.join(folder, 'train_seen_unseen_ids.npy'),
                                          os.path.join(folder, 'labels.npy'), folder)
        if okay:
            print("%s, %s and %s sequences in training, validation and test sets." % (
            len(train_v1_seq_idxs), len(seen_ids), len(unseen_seq_idxs)))
            print(len(set(train_v1_seq_idxs).intersection(set(unseen_seq_idxs))), len(set(train_v1_seq_idxs).intersection(set(seen_ids))))
            break

# create_splits_v1()

def get_split_stats(id_file):

    train_ids, val_ids, test_ids = np.load(id_file, allow_pickle=True)
    images = np.load(os.path.join(folder, 'img_cache' + str(min_len) + '.npy'), encoding='latin1')
    followings = np.load(os.path.join(folder, 'following_cache' + str(min_len) + '.npy'))

    train_img_ids = []
    for item in train_ids:
        train_img_ids.append(images[item])
        for i in range(len(followings[item])):
            train_img_ids.append(str(followings[item][i]))
    train_img_ids = list(set(train_img_ids))
    print("%s stories and %s unique frames in training split" % (len(train_ids), len(train_img_ids)))
    print('---------------------------------------------------------------------')

    val_img_ids = []
    unseen_stories = 0
    for item in val_ids:
        item_img_ids = []
        item_img_ids.append(images[item])
        for i in range(len(followings[item])):
            item_img_ids.append(str(followings[item][i]))
        if not any([id in train_img_ids for id in item_img_ids]):
            unseen_stories += 1
        val_img_ids.extend(item_img_ids)
    val_img_ids = list(set(val_img_ids))
    overlap_percent = float(len(set(val_img_ids).intersection(set(train_img_ids))))/len(val_img_ids)
    print("%s stories and %s unique frames in validation split" % (len(val_ids), len(val_img_ids)))
    print("%s stories with atleast one unseen frame in validation split" % unseen_stories)
    print("%s overlap in frames between training and validation" % overlap_percent)
    print('---------------------------------------------------------------------')

    test_img_ids = []
    unseen_stories = 0
    for item in test_ids:
        item_img_ids = []
        item_img_ids.append(images[item])
        for i in range(len(followings[item])):
            item_img_ids.append(str(followings[item][i]))
        if not any([id in train_img_ids for id in item_img_ids]):
            unseen_stories += 1
        test_img_ids.extend(item_img_ids)
    test_img_ids = list(set(test_img_ids))
    overlap_percent = float(len(set(test_img_ids).intersection(set(train_img_ids))))/len(test_img_ids)
    overlap_percent_val = float(len(set(test_img_ids).intersection(set(val_img_ids)))) / len(test_img_ids)
    print("%s stories and %s unique frames in validation split" % (len(test_ids), len(test_img_ids)))
    print("%s stories with atleast one unseen frame in validation split" % unseen_stories)
    print("%s overlap in frames between training and test" % overlap_percent)
    print("%s overlap in frames between validation and test" % overlap_percent_val)

get_split_stats('../pororo_png/train_seen_unseen_ids.npy')






