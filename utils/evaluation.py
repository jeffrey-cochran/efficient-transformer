from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# External imports
import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys

#
# Local imports
from utils.constants import bad_endings, IMAGE_ROOT

#
# Load coco-caption if available
try:
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    print("Warning: coco-caption not available")

#
# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ""
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + " "
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        if int(os.getenv("REMOVE_BAD_ENDINGS", "0")):
            flag = 0
            words = txt.split(" ")
            for j in range(len(words)):
                if words[-j - 1] not in bad_endings:
                    flag = -j
                    break
            txt = " ".join(words[0 : len(words) + flag])
        out.append(txt.replace("@@ ", ""))
    return out


#
# Detremine how many sentences end with an inappropriate word.
# SEE: utils.constants.bad_endings
def count_bad(sen):
    sen = sen.split(" ")
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


#
# Load annotation file
def getCOCO(dataset):
    annFile = "coco-caption/annotations/captions_val2014.json"
    return COCO(annFile)


def language_eval(dataset, preds, preds_n, job_id, split, eval_oracle=False):

    #
    # create output dictionary
    out = {}

    #
    # Diversity not implemented
    # ===========
    # if len(preds_n) > 0:
    #     # vocab size and novel sentences
    #     if "coco" in dataset:
    #         dataset_file = "data/dataset_coco.json"
    #     elif "flickr30k" in dataset or "f30k" in dataset:
    #         dataset_file = "data/dataset_flickr30k.json"
    #     training_sentences = set(
    #         [
    #             " ".join(__["tokens"])
    #             for _ in json.load(open(dataset_file))["images"]
    #             if not _["split"] in ["val", "test"]
    #             for __ in _["sentences"]
    #         ]
    #     )
    #     generated_sentences = set([_["caption"] for _ in preds_n])
    #     novels = generated_sentences - training_sentences
    #     out["novel_sentences"] = float(len(novels)) / len(preds_n)
    #     tmp = [_.split() for _ in generated_sentences]
    #     words = []
    #     for _ in tmp:
    #         words += _
    #     out["vocab_size"] = len(set(words))

    #
    # Set cache path
    cache_path = os.path.join("eval_results/", f".cache_{job_id}_{split}.json")

    #
    # Extract image ids in current data set
    coco = getCOCO(dataset)
    image_ids = coco.getImgIds()

    #
    # Filter results to only those in MSCOCO validation set
    filtered_predictions = [p for p in preds if p["image_id"] in image_ids]
    num_filtered_predictions = float(len(filtered_predictions))
    num_predictions = float(len(preds))

    #
    # Save predictions
    mean_perplexity = (
        sum([p["perplexity"] for p in filtered_predictions]) / num_filtered_predictions
    )
    mean_entropy = (
        sum([p["entropy"] for p in filtered_predictions]) / num_filtered_predictions
    )
    print(f"using {num_filtered_predictions}/{num_predictions} predictions")
    json.dump(
        filtered_predictions, open(cache_path, "w")
    )  # serialize to temporary json file. Sigh, COCO API...

    #
    # Evaluate captions
    # NOTE: loadRes() API call requires a json file, hence the above comment
    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()

    #
    # Compile results so far
    out["perplexity"] = mean_perplexity
    out["entropy"] = mean_entropy
    #
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    #
    # Record SPICE scores??
    imgToEval = cocoEval.imgToEval
    for k in list(imgToEval.values())[0]["SPICE"].keys():
        if k != "All":
            out["SPICE_" + k] = np.array(
                [v["SPICE"][k]["f"] for v in imgToEval.values()]
            )
            out["SPICE_" + k] = (
                out["SPICE_" + k][out["SPICE_" + k] == out["SPICE_" + k]]
            ).mean()
    #
    # Overwrite caption or set?
    for p in filtered_predictions:
        image_id, caption = p["image_id"], p["caption"]
        imgToEval[image_id]["caption"] = caption

    #
    # Diverse sampling not implemented
    # ==================
    # if len(preds_n) > 0:
    #     import eval_multi

    #     cache_path_n = os.path.join(
    #         "eval_results/", ".cache_" + job_id + "_" + split + "_n.json"
    #     )
    #     allspice = eval_multi.eval_allspice(dataset, preds_n, job_id, split)
    #     out.update(allspice["overall"])
    #     div_stats = eval_multi.eval_div_stats(dataset, preds_n, job_id, split)
    #     out.update(div_stats["overall"])
    #     if eval_oracle:
    #         oracle = eval_multi.eval_oracle(dataset, preds_n, job_id, split)
    #         out.update(oracle["overall"])
    #     else:
    #         oracle = None
    #     self_cider = eval_multi.eval_self_cider(dataset, preds_n, job_id, split)
    #     out.update(self_cider["overall"])
    #     with open(cache_path_n, "w") as outfile:
    #         json.dump(
    #             {
    #                 "allspice": allspice,
    #                 "div_stats": div_stats,
    #                 "oracle": oracle,
    #                 "self_cider": self_cider,
    #             },
    #             outfile,
    #         )

    #
    # Fraction of captions that have illegal endings
    # SEE: bad_endings in /utils/constants.py
    num_bad_endings = sum([count_bad(_["caption"]) for _ in filtered_predictions])
    out["bad_count_rate"] = num_bad_endings / num_filtered_predictions

    #
    # Write evaluation results to json
    outfile_path = os.path.join("eval_results/", f"{job_id}_{split}.json")
    with open(outfile_path, "w") as outfile:
        json.dump({"overall": out, "imgToEval": imgToEval}, outfile)
    #
    return out


def eval_split(
    model,
    crit,
    loader,
    verbose=True,
    verbose_beam=False,
    verbose_loss=True,
    num_images=-1,
    split="val",
    lang_eval=False,
    dataset="coco",
    beam_size=1,
    sample_n=1,
    remove_bad_endings=False,
    dump_path=False,
    dump_images=False,
    job_id="FUN_TIME",
):
    #
    # Use this nasty way to make other code clean since it's a global configuration
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)

    #
    # Make sure in the evaluation mode
    model.eval()

    #
    # Point iterator to validation data
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = []  # when sample_n > 1
    while True:
        #
        # len(data[]) is batch size
        data = loader.get_batch(split)
        n = n + len(data["infos"])

        #
        # If data is labeled and writing loss verbosely
        if data.get("labels", None) is not None and verbose_loss:
            #
            # Load into gpu memory, parallelize
            tmp = [
                data["fc_feats"],
                data["att_feats"],
                data["labels"],
                data["masks"],
                data["att_masks"],
            ]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            #
            # Compute average loss
            with torch.no_grad():
                loss = crit(
                    model(fc_feats, att_feats, labels, att_masks),
                    labels[..., 1:],
                    masks[..., 1:],
                ).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        #
        # Make sure cuda() is called if previous if(condition) failed
        tmp = [data["fc_feats"], data["att_feats"], data["att_masks"]]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp

        #
        # Only leave one feature for each image, in case duplicate sample
        with torch.no_grad():
            #
            # Generate sequence with current model
            seq, seq_logprobs = model.sample(fc_feats, att_feats, att_masks)
            seq = seq.data

            #
            # Compute entropy and perplexity of generated sequence
            entropy = -(F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / (
                (seq > 0).float().sum(1) + 1
            )
            perplexity = -seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / (
                (seq > 0).float().sum(1) + 1
            )

        #
        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(fc_feats.shape[0]):
                print(
                    "\n".join(
                        [
                            decode_sequence(loader.get_vocab(), _["seq"].unsqueeze(0))[
                                0
                            ]
                            for _ in model.done_beams[i]
                        ]
                    )
                )
                print("--" * 10)
        sents = decode_sequence(loader.get_vocab(), seq)

        #
        # Iterate over generated sentences
        for k, sent in enumerate(sents):
            entry = {
                "image_id": data["infos"][k]["id"],
                "caption": sent,
                "perplexity": perplexity[k].item(),
                "entropy": entropy[k].item(),
            }
            if dump_path:
                entry["file_name"] = data["infos"][k]["file_path"]
            predictions.append(entry)
            if dump_images:
                #
                # Dump the raw image to vis/ folder
                img_path_k = os.path.join(IMAGE_ROOT, data["infos"][k]["file_path"])
                img_destination_k = f"vis/imgs/{len(predictions)}.jpg"
                cmd = f'cp "{img_path_k}" {img_destination_k}'  # bit gross
                print(f"COPYING IMAGES......{cmd}")
                os.system(cmd)

            if verbose:
                print(f'image {entry["image_id"]}: {entry["caption"]}')

        #
        # Diverse sampling not implemented
        # ==================
        # if sample_n > 1:
        #     eval_split_n(
        #         model,
        #         n_predictions,
        #         loader,
        #         [fc_feats, att_feats, att_masks, data],
        #         eval_kwargs,
        #     )

        #
        # Iterator position over images
        current_image_index = data["bounds"]["it_pos_now"]
        num_images_evaluate = data["bounds"]["it_max"]
        if num_images != -1:
            num_images_evaluate = min(num_images_evaluate, num_images)
        else:
            num_images = num_images_evaluate

        #
        # This seems toatlly unnecessary...
        for i in range(n - num_images_evaluate):
            predictions.pop()

        if verbose:
            print(
                f"evaluating validation preformance... {current_image_index-1}/{num_images_evaluate} ({loss})"
            )

        if num_images >= 0 and n >= num_images:
            break

    #
    # DIVERSITY NOT IMPLEMENTED
    # ===============
    # if len(n_predictions) > 0 and "perplexity" in n_predictions[0]:
    #     n_predictions = sorted(n_predictions, key=lambda x: x["perplexity"])

    #
    # Save eval results
    if not os.path.isdir("eval_results"):
        os.mkdir("eval_results")
    #
    torch.save(
        (predictions, n_predictions),
        os.path.join("eval_results/", f".saved_pred_{job_id}_{split}.pth"),
    )
    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(
            dataset, predictions, n_predictions, job_id, split, eval_oracle=False
        )

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats


# Only run when sample_n > 0
# def eval_split_n(model, n_predictions, loader, input_data, eval_kwargs={}):
#     verbose = eval_kwargs.get("verbose", True)
#     beam_size = eval_kwargs.get("beam_size", 1)
#     sample_n = eval_kwargs.get("sample_n", 1)
#     sample_n_method = eval_kwargs.get("sample_n_method", "sample")

#     fc_feats, att_feats, att_masks, data = input_data

#     tmp_eval_kwargs = eval_kwargs.copy()
#     if sample_n_method == "bs":
#         # case 1 sample_n == beam size
#         tmp_eval_kwargs.update(
#             {"sample_n": 1, "beam_size": sample_n, "group_size": 1}
#         )  # randomness from softmax
#         with torch.no_grad():
#             model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode="sample")
#         for k in range(loader.batch_size):
#             _sents = utils.decode_sequence(
#                 loader.get_vocab(),
#                 torch.stack([model.done_beams[k][_]["seq"] for _ in range(sample_n)]),
#             )
#             for sent in _sents:
#                 entry = {"image_id": data["infos"][k]["id"], "caption": sent}
#                 n_predictions.append(entry)
#     # case 2 sample / gumbel / topk sampling/ nucleus sampling
#     elif (
#         sample_n_method == "sample"
#         or sample_n_method == "gumbel"
#         or sample_n_method.startswith("top")
#     ):
#         tmp_eval_kwargs.update(
#             {"sample_n": sample_n, "sample_method": sample_n_method, "beam_size": 1}
#         )  # randomness from sample
#         with torch.no_grad():
#             _seq, _sampleLogprobs = model(
#                 fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode="sample"
#             )
#         _sents = utils.decode_sequence(loader.get_vocab(), _seq)
#         _perplexity = -_sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(
#             1
#         ) / ((_seq > 0).float().sum(1) + 1)
#         for k, sent in enumerate(_sents):
#             entry = {
#                 "image_id": data["infos"][k // sample_n]["id"],
#                 "caption": sent,
#                 "perplexity": _perplexity[k].item(),
#             }
#             n_predictions.append(entry)
#     elif sample_n_method == "dbs":
#         # Use diverse beam search
#         tmp_eval_kwargs.update(
#             {"beam_size": sample_n * beam_size, "group_size": sample_n}
#         )  # randomness from softmax
#         with torch.no_grad():
#             model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode="sample")
#         for k in range(loader.batch_size):
#             _sents = utils.decode_sequence(
#                 loader.get_vocab(),
#                 torch.stack(
#                     [
#                         model.done_beams[k][_]["seq"]
#                         for _ in range(0, sample_n * beam_size, beam_size)
#                     ]
#                 ),
#             )
#             for sent in _sents:
#                 entry = {"image_id": data["infos"][k]["id"], "caption": sent}
#                 n_predictions.append(entry)
#     else:
#         tmp_eval_kwargs.update(
#             {
#                 "sample_method": sample_n_method[1:],
#                 "group_size": sample_n,
#                 "beam_size": 1,
#             }
#         )  # randomness from softmax
#         with torch.no_grad():
#             _seq, _sampleLogprobs = model(
#                 fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode="sample"
#             )
#         _sents = utils.decode_sequence(loader.get_vocab(), _seq)
#         for k, sent in enumerate(_sents):
#             entry = {"image_id": data["infos"][k // sample_n]["id"], "caption": sent}
#             n_predictions.append(entry)
#     if verbose:
#         for entry in sorted(
#             n_predictions[-loader.batch_size * sample_n :], key=lambda x: x["image_id"]
#         ):
#             print("image %s: %s" % (entry["image_id"], entry["caption"]))
