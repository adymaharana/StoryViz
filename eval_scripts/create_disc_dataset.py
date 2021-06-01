import os
import random
import numpy as np
from tqdm import tqdm
import json
import time
import torch
import torchvision.utils as vutils
import PIL

def sample_image(im):
    shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
    video_len = int(longer/shorter)
    se = np.random.randint(0,video_len, 1)[0]
    #print(se*shorter, shorter, (se+1)*shorter)
    return np.array(im.crop((0, se * shorter, shorter, (se+1)*shorter)))

def prepare_discriminative_dataset_v0(data_dir):
    # The last frame is considered for discrimination

    min_len = 4
    images = np.load(os.path.join(data_dir, 'img_cache' + str(min_len) + '.npy'), encoding='latin1')
    followings = np.load(os.path.join(data_dir, 'following_cache' + str(min_len) + '.npy'))
    train_id, val_id, test_id = np.load(os.path.join(data_dir, 'train_seen_unseen_ids.npy'), allow_pickle=True)
    labels = np.load(os.path.join(data_dir, 'labels.npy'), allow_pickle=True, encoding='latin1').item()

    train_img_ids = [str(images[item]).replace('.png', '')[2:-1] for item in train_id]
    disc_dataset = {}

    for ids in [val_id, test_id]:
        skipped = 0
        for i, item in tqdm(enumerate(ids)):
            lists = [images[item]]
            for i in range(len(followings[item])):
                lists.append(str(followings[item][i]))

            final_img = lists[-1]
            id = str(final_img).replace('.png', '')[2:-1]
            _, ep_id, _ = id.split('/')
            label = labels[id]
            start = time.time()
            choice_ids = [key for key in train_img_ids if (np.array_equal(label, labels[key]) and key.split('/')[1] != ep_id)]
            end = time.time()
            # print(end-start)

            # if less than 4 choices exist, select a character in the query frame and get choices for that
            if len(choice_ids) <= 4:
                character_idx = random.choice([idx for idx in range(label.shape[0]) if label[idx] == 1])
                choice_ids = [key for key, val in labels.items() if
                              val[character_idx] == 1 and key.split('/')[1] != ep_id]
            # if still less than 4 choices, take whatever we have
            if len(choice_ids) <= 4:
                print(item, label, len(choice_ids))
                choices = choice_ids
            else:
                choices = [os.path.join(data_dir, choice_id + '.png') for choice_id in random.choices(choice_ids, k=4)]

            tgt_path = os.path.join(data_dir, id + '.png')


            disc_dataset[int(item)] = {
                'target': tgt_path,
                'choices': choices
            }
            # print(type(item))
            # break
        print('Skipped %s samples' % skipped)

    # print(disc_dataset)
    with open(os.path.join(data_dir, 'disc_dataset.json'), 'w') as f:
        json.dump(disc_dataset, f)


def prepare_discriminative_dataset_v1(data_dir):
    # The first frame is considered for discrimination

    min_len = 4
    images = np.load(os.path.join(data_dir, 'img_cache' + str(min_len) + '.npy'), encoding='latin1')
    followings = np.load(os.path.join(data_dir, 'following_cache' + str(min_len) + '.npy'))
    train_id, val_id, test_id = np.load(os.path.join(data_dir, 'train_seen_unseen_ids.npy'), allow_pickle=True)
    labels = np.load(os.path.join(data_dir, 'labels.npy'), allow_pickle=True, encoding='latin1').item()

    train_img_ids = [str(images[item]).replace('.png', '')[2:-1] for item in train_id]
    disc_dataset = {}

    for ids in [val_id, test_id]:
        skipped = 0
        for i, item in tqdm(enumerate(ids)):
            lists = [images[item]]
            for i in range(len(followings[item])):
                lists.append(str(followings[item][i]))

            first_img = lists[0]
            id = str(first_img).replace('.png', '')[2:-1]
            _, ep_id, _ = id.split('/')
            label = labels[id]
            start = time.time()
            choice_ids = [key for key in train_img_ids if (np.array_equal(label, labels[key]) and key.split('/')[1] != ep_id)]
            end = time.time()
            # print(end-start)

            # if less than 4 choices exist, select a character in the query frame and get choices for that
            if len(choice_ids) <= 4:
                character_idx = random.choice([idx for idx in range(label.shape[0]) if label[idx] == 1])
                choice_ids = [key for key, val in labels.items() if
                              val[character_idx] == 1 and key.split('/')[1] != ep_id]
            # if still less than 4 choices, take whatever we have
            if len(choice_ids) <= 4:
                print(item, label, len(choice_ids))
                choices = choice_ids
            else:
                choices = [os.path.join(data_dir, choice_id + '.png') for choice_id in random.choices(choice_ids, k=4)]
            tgt_path = os.path.join(data_dir, id + '.png')

            disc_dataset[int(item)] = {
                'target': tgt_path,
                'choices': choices
            }
            # print(type(item))
            # break
        print('Skipped %s samples' % skipped)

    # print(disc_dataset)
    with open(os.path.join(data_dir, 'disc_dataset_first_frame.json'), 'w') as f:
        json.dump(disc_dataset, f)

# prepare_discriminative_dataset_v1('../pororo_png/')

def prepare_human_evaluation_by_pos(dir_path):

    images = np.load(os.path.join(dir_path, 'img_cache4.npy'))
    followings = np.load(os.path.join(dir_path, 'following_cache4.npy'))
    descriptions_original = np.load(os.path.join(dir_path, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
    train_id, val_id, test_id = np.load(os.path.join(dir_path, 'train_seen_unseen_ids.npy'), allow_pickle=True)

    with open(os.path.join(dir_path, 'disc_dataset_first_frame.json'), 'r') as f:
        disc = json.load(f)

    print("%s samples in Disc dataset" % len(disc))
    idxs = random.choices(range(len(disc)), k=20)

    all_images = []
    correct_idxs = []
    all_descriptions = []
    human_eval_idxs = []
    for i, (key, val) in enumerate(disc.items()):
        if i not in idxs:
            continue

        test_id = int(key)
        lists = [images[test_id]]
        for i in range(len(followings[test_id])):
            lists.append(followings[test_id][i])

        tgt_path = val['target']
        assert tgt_path == os.path.join(dir_path, str(lists[0])[2:-1]), (tgt_path, os.path.join(dir_path, str(lists[0])[2:-1]))
        choices = val['choices'] + [tgt_path]

        if len(choices) != 5:
            continue

        human_eval_idxs.append(test_id)

        descriptions = []
        # frame_images = []
        for v in lists[0:1]:
            id = str(v).replace('.png', '')[2:-1]
            descriptions.append(random.choice(descriptions_original[id]))
        #     frame_images.append(torch.tensor(sample_image(PIL.Image.open(os.path.join(dir_path, id + '.png')))))

        all_descriptions.append(descriptions)
        random.shuffle(choices)
        correct_idx = [i for i in range(len(choices)) if choices[i] == tgt_path]
        correct_idxs.append(correct_idx)
        choice_images = [torch.tensor(sample_image(PIL.Image.open(os.path.join(dir_path, c)))) for c in choices]
        # all_images.append(frame_images[:-1] + choice_images)
        all_images.append(choice_images)

    all_images_2 = []
    for i in range(len(all_images)):
        all_images_2.append(vutils.make_grid(torch.cat([x.permute(2, 0, 1).unsqueeze(0) for x in all_images[i]], dim=0), 5))
        print(all_images_2[0][0])
        print(all_images_2[0].shape)
    all_images_2= vutils.make_grid(all_images_2, 1)
    print(all_images_2.shape)
    # all_images_2 = images_to_numpy(all_images_2)

    output = PIL.Image.fromarray(np.uint8(all_images_2.cpu().numpy().transpose(1, 2, 0)))
    output.save('human_eval_first_frame.png')
    np.save('human_eval_first_frame_labels.npy', np.array(correct_idxs))
    np.save('human_eval_first_frame_ids.npy', np.array(human_eval_idxs))

    with open('human_eval_first_frame_descriptions.txt', 'w') as f:
        for i, des in enumerate(all_descriptions):
            f.write('-----------------%s------------------\n' % i)
            for d in des:
                f.write(d + '\n')
            f.write("Label: " + str(correct_idxs[i]) + '\n')
            f.write('\n')

prepare_human_evaluation_by_pos('../pororo_png/')