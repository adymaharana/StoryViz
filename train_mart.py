import torch
from tqdm import tqdm
import argparse
import random
import os
import numpy as np
from easydict import EasyDict as EDict
from mart.data_loader import get_loader, prepare_batch_inputs
import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
from mart.optimization import BertAdam, EMA
import time
import math
import torch.nn as nn
from torchvision import transforms, models
from mart.data_loader import MyInceptionFeatureExtractor
from mart.recurrent import RecursiveTransformer


def cal_performance(pred, gold):
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(-1)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum().item()
    return n_correct

def extract_img_features(feature_extractor, input_images_list, total_seq_len, device):
    input_imgs = torch.cat(input_images_list, dim=0).to(device)
    bsz = input_images_list[0].shape[0]
    # print(input_imgs.shape)
    features = feature_extractor(input_imgs).permute(0, 2, 3, 1).view(-1, 64, 2048)
    # print(features.shape)
    outputs = [torch.zeros(bsz, total_seq_len, 2048).to(device) for _ in range(len(input_images_list))]
    for i in range(len(input_images_list)):
        outputs[i][:, 1:65, :] = features[i*bsz:(i+1)*bsz, :, :]
    return outputs

def train_epoch(model, training_data_loader, optimizer, device, opt, epoch, feature_extractor):
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    torch.autograd.set_detect_anomaly(True)
    for batch_idx, batch in tqdm(enumerate(training_data_loader), mininterval=2,
                                 desc="  Training =>", total=len(training_data_loader)):
        niter = epoch * len(training_data_loader) + batch_idx
        total_seq_len = training_data_loader.dataset.max_v_len + training_data_loader.dataset.max_t_len

        # TODO: extract features
        if opt.debug:
            print([b["image"].shape for b in batch])
            total_seq_len = training_data_loader.dataset.max_v_len + training_data_loader.dataset.max_t_len
            for i in range(5):
                batch[i]["video_feature"] = torch.tensor(torch.zeros((opt.batch_size, total_seq_len, opt.video_feature_size)))
        else:
            video_features_list = extract_img_features(feature_extractor, [b["image"] for b in batch], total_seq_len, device)
            for i in range(5):
                batch[i]["video_feature"] = video_features_list[i]

        # prepare data
        batched_data = [prepare_batch_inputs(step_data, bsz=opt.batch_size, device=device, non_blocking=opt.pin_memory) for step_data in batch]
        input_ids_list = [e["input_ids"] for e in batched_data]
        video_features_list = [e["video_feature"] for e in batched_data]
        input_masks_list = [e["input_mask"] for e in batched_data]
        token_type_ids_list = [e["token_type_ids"] for e in batched_data]
        input_labels_list = [e["input_labels"] for e in batched_data]

        # forward & backward
        optimizer.zero_grad()
        loss, pred_scores_list = model(input_ids_list, video_features_list,
                                       input_masks_list, token_type_ids_list, input_labels_list)


        loss.backward()
        if opt.grad_clip != -1:  # enable, -1 == disable
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        # keep logs
        n_correct = 0
        n_word = 0
        for pred, gold in zip(pred_scores_list, input_labels_list):
            n_correct += cal_performance(pred, gold)
            valid_label_mask = gold.ne(-1)
            n_word += valid_label_mask.sum().item()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info("[Training]  iteration loss: {loss: 8.5f}, accuracy: {acc:3.3f} %"
                        .format(loss=loss.item(), acc=100 * float(n_correct)/n_word))

        if opt.debug:
            break

    torch.autograd.set_detect_anomaly(False)
    loss_per_word = 1.0 * total_loss / n_word_total
    accuracy = 1.0 * n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data_loader, device, opt, feature_extractor):
    """The same setting as training, where ground-truth word x_{t-1}
    is used to predict next word x_{t}, not realistic for real inference"""
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(validation_data_loader, mininterval=2, desc="  Validation =>"):

            # TODO: extract features
            if opt.debug:
                total_seq_len = validation_data_loader.dataset.max_v_len + validation_data_loader.dataset.max_t_len
                for i in range(5):
                    batch[i]["video_feature"] = torch.tensor(
                        torch.zeros((opt.val_batch_size, total_seq_len, opt.video_feature_size)))
            else:
                total_seq_len = validation_data_loader.dataset.max_v_len + validation_data_loader.dataset.max_t_len
                video_features_list = extract_img_features(feature_extractor, [b["image"] for b in batch],
                                                           total_seq_len, device)
                for i in range(5):
                    batch[i]["video_feature"] = video_features_list[i]

            # prepare data
            batched_data = [prepare_batch_inputs(step_data, opt.val_batch_size, device=device, non_blocking=opt.pin_memory)
                            for step_data in batch]
            input_ids_list = [e["input_ids"] for e in batched_data]
            video_features_list = [e["video_feature"] for e in batched_data]
            input_masks_list = [e["input_mask"] for e in batched_data]
            token_type_ids_list = [e["token_type_ids"] for e in batched_data]
            input_labels_list = [e["input_labels"] for e in batched_data]

            loss, pred_scores_list = model(input_ids_list, video_features_list,
                                           input_masks_list, token_type_ids_list, input_labels_list)


            # keep logs
            n_correct = 0
            n_word = 0
            bleu = 0
            for pred, gold in zip(pred_scores_list, input_labels_list):
                n_correct += cal_performance(pred, gold)
                valid_label_mask = gold.ne(-1)
                n_word += valid_label_mask.sum().item()

            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

            if opt.debug:
                break

    loss_per_word = 1.0 * total_loss / n_word_total
    accuracy = 1.0 * n_word_correct / n_word_total
    return loss_per_word, accuracy


def train(model, training_data_loader, validation_data_loader, device, opt, feature_extractor, test_data_loader=None):

    model = model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    num_train_optimization_steps = len(training_data_loader) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.lr,
                         warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps,
                         schedule="warmup_linear")

    for epoch_i in range(opt.n_epoch):
        logger.info("[Epoch {}]".format(epoch_i))

        # schedule sampling prob update, TODO not implemented yet

        start = time.time()
        train_loss, train_acc = train_epoch(
            model, training_data_loader, optimizer, device, opt, epoch_i, feature_extractor)
        logger.info("[Training]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min"
                    .format(ppl=math.exp(min(train_loss, 100)), acc=100 * train_acc,
                            elapse=(time.time() - start) / 60.))
        niter = (epoch_i + 1) * len(training_data_loader)  # number of bart

        checkpoint = {
            "model": model.state_dict(),  # EMA model
            "model_cfg": model.config,
            "opt": opt,
            "epoch": epoch_i}

        eval_loss, eval_acc = eval_epoch(model, validation_data_loader, device, opt, feature_extractor)
        logger.info("[Validation]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %"
                    .format(ppl=math.exp(min(eval_loss, 100)), acc=100 * eval_acc))

        if epoch_i %5 == 0 and test_data_loader is not None:
            test_loss, test_acc = eval_epoch(model, test_data_loader, device, opt, feature_extractor)
            logger.info("[Test]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %"
                        .format(ppl=math.exp(min(test_loss, 100)), acc=100 * test_acc))

        model_name = opt.save_model + "_e{e}_b{b}.chkpt".format(
            e=epoch_i, b=round(eval_acc * 100, 2))
        torch.save(checkpoint, model_name)


def init_feature_extractor(debug=False):

    if debug:
        return None

    model_ft = models.inception_v3(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluate mode

    feature_extractor = MyInceptionFeatureExtractor(model_ft).to(device)
    return feature_extractor

def get_args():

    """parse and preprocess cmd line args"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dset_name", type=str, default="pororo", choices=["pororo"],
                        help="Name of the dataset, will affect data loader, evaluation, etc")

    # model config
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, help="number of words in the vocabulary")
    parser.add_argument("--word_vec_size", type=int, default=300)
    parser.add_argument("--video_feature_size", type=int, default=2048, help="2048 appearance")
    parser.add_argument("--max_v_len", type=int, default=64, help="max length of video feature")
    parser.add_argument("--max_t_len", type=int, default=25,
                        help="max length of text (sentence or paragraph), 30 for anet, 20 for yc2")
    parser.add_argument("--max_n_sen", type=int, default=6,
                        help="for recurrent, max number of sentences, 6 for anet, 10 for yc2")
    parser.add_argument("--n_memory_cells", type=int, default=1, help="number of memory cells in each layer")
    parser.add_argument("--type_vocab_size", type=int, default=2, help="video as 0, text as 1")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of transformer layers")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--memory_dropout_prob", type=float, default=0.1)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--raw_glove_path", type=str, default='../../data/glove.840B.300d.txt', help="raw GloVe vectors")
    parser.add_argument("--vocab_glove_path", type=str, default=None, help="extracted GloVe vectors")
    parser.add_argument("--freeze_glove", action="store_true", help="do not train GloVe vectors")
    parser.add_argument("--share_wd_cls_weight", action="store_true",
                        help="share weight matrix of the word embedding with the final classifier, ")

    # training config -- learning rate
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--grad_clip", type=float, default=1, help="clip gradient, -1 == disable")
    parser.add_argument("--ema_decay", default=0.9999, type=float,
                        help="Use exponential moving average at training, float in (0, 1) and -1: do not use.  "
                             "ema_param = new_param * ema_decay + (1-ema_decay) * last_param")

    parser.add_argument("--data_dir", required=True, help="dir containing the splits data files")
    parser.add_argument("--word2idx_path", type=str, default="./cache/word2idx.json")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Use soft target instead of one-hot hard target")
    parser.add_argument("--n_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max_es_cnt", type=int, default=10,
                        help="stop if the model is not improving for max_es_cnt max_es_cnt")
    parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
    parser.add_argument("--val_batch_size", type=int, default=50, help="inference batch size")

    parser.add_argument("--use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("--beam_size", type=int, default=2, help="beam size")
    parser.add_argument("--n_best", type=int, default=1, help="stop searching when get n_best from beam search")

    # others
    parser.add_argument("---num_workers", type=int, default=8,
                        help="num subprocesses used to load the data, 0: use main process")
    parser.add_argument("--save_model", default="model")
    parser.add_argument("--save_mode", type=str, choices=["all", "best"], default="best",
                        help="all: save models at each epoch; best: only save the best model")
    parser.add_argument("--res_root_dir", type=str, default='./out/')
    parser.add_argument("--no_cuda", action="store_true", help="run on cpu")
    parser.add_argument("--seed", default=2019, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval_tool_dir", type=str, default="./densevid_eval")

    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Don't use pin_memory=True for dataloader. "
                             "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.pin_memory = not opt.no_pin_memory

    model_type = 'mart'
    # make paths
    opt.res_dir = os.path.join(
        opt.res_root_dir, "_".join([opt.dset_name, model_type, time.strftime("%Y_%m_%d_%H_%M_%S")]))
    if opt.debug:
        opt.res_dir = "debug_" + opt.res_dir

    if os.path.exists(opt.res_dir) and os.listdir(opt.res_dir):
        raise ValueError("File exists {}".format(opt.res_dir))
    elif not os.path.exists(opt.res_dir):
        os.makedirs(opt.res_dir)

    opt.log = os.path.join(opt.res_dir, opt.save_model)
    opt.save_model = os.path.join(opt.res_dir, opt.save_model)

    if opt.share_wd_cls_weight:
        assert opt.word_vec_size == opt.hidden_size, \
            "hidden size has to be the same as word embedding size when " \
            "sharing the word embedding weight and the final classifier weight"
    return opt


def main():

    opt = get_args()

    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    vocab_threshold = 5
    # hardcoded for InceptionNet as feature extractor
    im_input_size = 299
    vocab_from_file = True

    transform_train = transforms.Compose([
        transforms.Resize(im_input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(im_input_size),
        transforms.CenterCrop(im_input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    vocab_file = os.path.join(opt.data_dir, 'videocap_vocab.pkl')
    train_loader = get_loader(transform=transform_train,
                              data_dir=opt.data_dir,
                              mode='train',
                              batch_size=opt.batch_size,
                              vocab_threshold=vocab_threshold,
                              vocab_from_file=vocab_from_file,
                              vocab_file=vocab_file)

    if opt.debug:
        # print(train_loader.dataset[0][0]["image"])
        pass

    # add 10 at max_n_sen to make the inference stage use all the segments
    val_loader = get_loader(transform=transform_val,
                            data_dir=opt.data_dir,
                            mode='val',
                            batch_size=opt.val_batch_size,
                            vocab_threshold=vocab_threshold,
                            vocab_from_file=vocab_from_file,
                            vocab_file=vocab_file)

    opt.vocab_size = train_loader.dataset.vocab_size

    if opt.vocab_glove_path is None:
        opt.vocab_glove_path = os.path.join(opt.data_dir, 'mart_glove_embeddings.mat')
    train_loader.dataset.vocab.extract_glove(opt.raw_glove_path, opt.vocab_glove_path)

    opt.max_t_len = train_loader.dataset.max_t_len
    opt.max_v_len = train_loader.dataset.max_v_len

    device = torch.device("cuda" if opt.cuda else "cpu")
    rt_config = EDict(
        hidden_size=opt.hidden_size,
        intermediate_size=opt.intermediate_size,  # after each self attention
        vocab_size=opt.vocab_size,  # get from word2idx
        word_vec_size=opt.word_vec_size,
        padding_idx=train_loader.dataset.vocab.word2idx[train_loader.dataset.vocab.pad_word],
        video_feature_size=opt.video_feature_size,
        max_position_embeddings=opt.max_v_len + opt.max_t_len,  # get from max_seq_len
        max_v_len=opt.max_v_len,  # max length of the videos
        max_t_len=opt.max_t_len,  # max length of the text
        type_vocab_size=opt.type_vocab_size,
        layer_norm_eps=opt.layer_norm_eps,  # bert layernorm
        hidden_dropout_prob=opt.hidden_dropout_prob,  # applies everywhere except attention
        num_hidden_layers=opt.num_hidden_layers,  # number of transformer layers
        num_attention_heads=opt.num_attention_heads,
        attention_probs_dropout_prob=opt.attention_probs_dropout_prob,  # applies only to self attention
        n_memory_cells=opt.n_memory_cells,  # memory size will be (n_memory_cells, D)
        memory_dropout_prob=opt.memory_dropout_prob,
        initializer_range=opt.initializer_range,
        label_smoothing=opt.label_smoothing,
        share_wd_cls_weight=opt.share_wd_cls_weight
    )

    model = RecursiveTransformer(rt_config)

    if opt.vocab_glove_path is not None:
        if hasattr(model, "embeddings"):
            logger.info("Load GloVe as word embedding")
            model.embeddings.set_pretrained_embedding(
                torch.from_numpy(torch.load(opt.vocab_glove_path)).float(), freeze=opt.freeze_glove)
        else:
            logger.warning("This model has no embeddings, cannot load glove vectors into the model")

    feature_extractor = init_feature_extractor(opt.debug)

    train(model, train_loader, val_loader, device, opt, feature_extractor)

if __name__ == "__main__":
    main()