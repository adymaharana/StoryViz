""" Translate input text with trained model. """
import os, json
import torch
import argparse
from tqdm import tqdm
import random
import numpy as np
import subprocess
from torchvision import transforms, models

from mart.translator import Translator
from mart.data_loader import get_loader, prepare_batch_inputs, MyInceptionFeatureExtractor
from mart.utils import save_json

def init_feature_extractor(debug=False):

    if debug:
        return None

    model_ft = models.inception_v3(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False

    # # Handle the auxilary net
    # num_ftrs = model_ft.AuxLogits.fc.in_features
    # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # # Handle the primary net
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, num_classes)
    # input_size = 299
    #
    # model_ft.load_state_dict(
    #     torch.load('/ssd-playpen/home/adyasha/projects/StoryGAN/classifier/old_models/epoch-49.pt'))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluate mode

    feature_extractor = MyInceptionFeatureExtractor(model_ft).to(device)
    return feature_extractor


def sort_res(res_dict):
    """res_dict: the submission json entry `results`"""
    final_res_dict = {}
    for k, v in res_dict.items():
        final_res_dict[k] = sorted(v, key=lambda x: float(x["timestamp"][0]))
    return final_res_dict

def extract_img_features(feature_extractor, input_images_list, total_seq_len, device, debug=False):
    input_imgs = torch.cat(input_images_list, dim=0).to(device)
    bsz = input_images_list[0].shape[0]
    # print(input_imgs.shape)
    if debug:
        outputs = [torch.zeros(bsz, total_seq_len, 2048).to(device) for _ in range(len(input_images_list))]
    else:
        features = feature_extractor(input_imgs).permute(0, 2, 3, 1).view(-1, 64, 2048)
        # print(features.shape)
        outputs = [torch.zeros(bsz, total_seq_len, 2048).to(device) for _ in range(len(input_images_list))]
        for i in range(len(input_images_list)):
            outputs[i][:, 1:65, :] = features[i*bsz:(i+1)*bsz, :, :]
    return outputs

def convert_ids_to_sentence(ids, vocab, rm_padding=True, return_sentence_only=True):
    """A list of token ids"""
    PAD = 3
    IGNORE = -1
    EOS_TOKEN = '[EOS]'
    rm_padding = True if return_sentence_only else rm_padding
    if rm_padding:
        raw_words = [vocab.idx2word[wid] for wid in ids if wid not in [PAD, IGNORE]]
    else:
        raw_words = [vocab.idx2word[wid] for wid in ids if wid != IGNORE]

    # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
    if return_sentence_only:
        words = []
        for w in raw_words[1:]:  # no [BOS]
            if w != EOS_TOKEN:
                words.append(w)
            else:
                break
    else:
        words = raw_words
    return " ".join(words)

def run_translate(eval_data_loader, translator, opt, feature_extractor, device, vocab, debug=False):
    # submission template
    batch_res = {"version": "VERSION 1.0",
                 "results": [],
                 "external_data": {"used": "true", "details": "ay"}}
    for batch in tqdm(eval_data_loader, mininterval=2, desc="  - (Translate)"):

        # prepare data
        if opt.debug:
            print([b["image"].shape for b in batch])
            total_seq_len = eval_data_loader.dataset.max_v_len + eval_data_loader.dataset.max_t_len
            for i in range(5):
                batch[i]["video_feature"] = torch.tensor(torch.zeros((opt.val_batch_size, total_seq_len, opt.video_feature_size)))
        else:
            total_seq_len = eval_data_loader.dataset.max_v_len + eval_data_loader.dataset.max_t_len
            video_features_list = extract_img_features(feature_extractor, [b["image"] for b in batch], total_seq_len, device, debug=debug)
            for i in range(5):
                batch[i]["video_feature"] = video_features_list[i]

        # prepare data
        batched_data = [prepare_batch_inputs(step_data, bsz=opt.val_batch_size, device=device, non_blocking=opt.pin_memory) for step_data in batch]
        model_inputs = [
            [e["input_ids"] for e in batched_data],
            [e["video_feature"] for e in batched_data],
            [e["input_mask"] for e in batched_data],
            [e["token_type_ids"] for e in batched_data]
        ]
        dec_seq_list = translator.translate_batch(
            model_inputs, use_beam=opt.use_beam, recurrent=True, untied=False, xl=False)

        # print(len(dec_seq_list), len(dec_seq_list[0]))
        # print(len(batch[0]["caption"]), len(batch))

        step_size = 5
        # example_idx indicates which example is in the batch
        for example_idx in range(opt.val_batch_size):
            # step_idx or we can also call it sen_idx
            for step_idx, step_batch in enumerate(dec_seq_list[:step_size]):
                batch_res["results"].append({
                    "sentence": convert_ids_to_sentence(step_batch[example_idx].cpu().tolist(), vocab),
                    "gt_sentence": batch[step_idx]["caption"][example_idx]
                })

        if opt.debug:
            break

    # batch_res["results"] = sort_res(batch_res["results"])
    # print(batch_res)
    return batch_res


def main():
    parser = argparse.ArgumentParser(description="translate.py")

    parser.add_argument("--eval_mode", type=str, default="val",
                        choices=["val", "test"], help="evaluate on val/test set, yc2 only has val")
    parser.add_argument("--res_dir", required=True, help="path to dir containing model .pt file")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")

    # beam search configs
    parser.add_argument("--use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("--beam_size", type=int, default=2, help="beam size")
    parser.add_argument("--n_best", type=int, default=1, help="stop searching when get n_best from beam search")
    parser.add_argument("--min_sen_len", type=int, default=5, help="minimum length of the decoded sentences")
    parser.add_argument("--max_sen_len", type=int, default=30, help="maximum length of the decoded sentences")
    parser.add_argument("--block_ngram_repeat", type=int, default=0, help="block repetition of ngrams during decoding.")
    parser.add_argument("--length_penalty_name", default="none",
                        choices=["none", "wu", "avg"], help="length penalty to use.")
    parser.add_argument("--length_penalty_alpha", type=float, default=0.,
                        help="Google NMT length penalty parameter (higher = longer generation)")
    parser.add_argument("--eval_tool_dir", type=str, default="./densevid_eval")

    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", default=2019, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint_file", type=str)

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, default='')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    checkpoint = torch.load(os.path.join(opt.res_dir, opt.checkpoint_file))

    # add some of the train configs
    train_opt = checkpoint["opt"]  # EDict(load_json(os.path.join(opt.res_dir, "model.cfg.json")))
    for k in train_opt.__dict__:
        if k not in opt.__dict__:
            setattr(opt, k, getattr(train_opt, k))
    print("train_opt", train_opt)
    opt.val_batch_size = opt.batch_size

    decoding_strategy = "beam{}_lp_{}_la_{}".format(
        opt.beam_size, opt.length_penalty_name, opt.length_penalty_alpha) if opt.use_beam else "greedy"
    save_json(vars(opt),
              os.path.join(opt.res_dir, "{}_eval_cfg.json".format(decoding_strategy)),
              save_pretty=True)

    # if opt.dset_name == "anet":
    #     reference_files_map = {
    #         "val": [os.path.join(opt.data_dir, e) for e in
    #                 ["anet_entities_val_1_para.json", "anet_entities_val_2_para.json"]],
    #         "test": [os.path.join(opt.data_dir, e) for e in
    #                  ["anet_entities_test_1_para.json", "anet_entities_test_2_para.json"]]}
    # else:  # yc2
    #     reference_files_map = {"val": [os.path.join(opt.data_dir, "yc2_val_anet_format_para.json")]}

    # for eval_mode in opt.eval_splits:

    vocab_threshold = 5
    # hardcoded for InceptionNet as feature extractor
    im_input_size = 299
    vocab_from_file = True
    transform_val = transforms.Compose([
        transforms.Resize(im_input_size),
        transforms.CenterCrop(im_input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Start evaluating {}".format(opt.eval_mode))
    # add 10 at max_n_sen to make the inference stage use all the segments
    eval_data_loader = get_loader(transform=transform_val,
                                  data_dir=opt.data_dir,
                                  mode=opt.eval_mode,
                                  batch_size=opt.val_batch_size,
                                  vocab_threshold=vocab_threshold,
                                  vocab_from_file=vocab_from_file,
                                  pred_img_dir=opt.pred_dir,
                                  vocab_file=os.path.join(opt.data_dir, 'videocap_vocab.pkl'))
    # eval_references = reference_files_map[eval_mode]

    # setup model
    translator = Translator(opt, checkpoint)

    feature_extractor = init_feature_extractor(opt.debug)

    device = torch.device("cuda" if opt.cuda else "cpu")

    pred_file = os.path.join(opt.res_dir, "{}_pred_{}.json".format(decoding_strategy, opt.eval_mode))
    pred_file = os.path.abspath(pred_file)
    # if not os.path.exists(pred_file):
    json_res = run_translate(eval_data_loader, translator, opt,
                             feature_extractor, device, eval_data_loader.dataset.vocab)
    save_json(json_res, pred_file, save_pretty=True)
    # else:
    # print("Using existing prediction file at {}".format(pred_file))

    with open(pred_file, 'r') as f:
        d = json.load(f)
    hyps = []
    refs = []
    for res in d["results"]:
        hyps.append(res["sentence"])
        refs.append(res["gt_sentence"])
    with open(pred_file.replace(".json", "_hyps.txt"), "w") as f:
        f.write("\n".join(hyps))
    with open(pred_file.replace(".json", "_refs.txt"), "w") as f:
        f.write("\n".join(refs))

    # # BLEU Evaluation
    eval_command = ["nlg-eval", "--hypothesis=" + pred_file.replace(".json", "_hyps.txt"), "--references=" + pred_file.replace(".json", "_refs.txt")]
    subprocess.call(eval_command)

    # # COCO language evaluation
    # lang_file = pred_file.replace(".json", "_lang.json")
    # eval_command = ["python", "para-evaluate.py", "-s", pred_file, "-o", lang_file,
    #                 "-v", "-r"] + eval_references
    # subprocess.call(eval_command, cwd=opt.eval_tool_dir)
    #
    # # basic stats
    # stat_filepath = pred_file.replace(".json", "_stat.json")
    # eval_stat_cmd = ["python", "get_caption_stat.py", "-s", pred_file, "-r", eval_references[0],
    #                  "-o", stat_filepath, "-v"]
    # subprocess.call(eval_stat_cmd, cwd=opt.eval_tool_dir)
    #
    # # repetition evaluation
    # rep_filepath = pred_file.replace(".json", "_rep.json")
    # eval_rep_cmd = ["python", "evaluateRepetition.py", "-s", pred_file,
    #                 "-r", eval_references[0], "-o", rep_filepath]
    # subprocess.call(eval_rep_cmd, cwd=opt.eval_tool_dir)
    #
    # metric_filepaths = [lang_file, stat_filepath, rep_filepath]
    # all_metrics = merge_dicts([load_json(e) for e in metric_filepaths])
    # all_metrics_filepath = pred_file.replace(".json", "_all_metrics.json")
    # save_json(all_metrics, all_metrics_filepath, save_pretty=True)
    #
    # print("pred_file {} lang_file {}".format(pred_file, lang_file))
    # print("[Info] Finished {}.".format(eval_mode))


if __name__ == "__main__":
    main()