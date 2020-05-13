from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#
# External imports
from torch import arange as torch_arange, load as torch_load
from torch.nn import DataParallel
from torch.cuda import synchronize
from torch.utils.tensorboard import SummaryWriter
import skimage.io
import numpy as np
import time
from os.path import join, isfile
from six.moves import cPickle
from traceback import format_exc
from collections import defaultdict


#
# Local imports
from utils.config import CfgNode
from utils.constants import (
    checkpoint_path,
    REDUCE_LR,
    NOAM,
    CLIP_GRAD,
    CLIP_VALUE,
    gradient_clipping_functions,
)
from utils.data.DataLoader_def import DataLoader
from utils.data.Dataset_def import Dataset
from utils.data.HybridLoader_def import HybridLoader
from utils.data.CustomSampler_def import CustomSampler
from utils.evaluation import eval_split
from utils.loss.LossWrapper_def import LossWrapper
from utils.loss.rewards import init_scorer
from utils.misc import pickle_load, save_checkpoint, BAR
from utils.optimization.NoamOpt_def import NoamOpt, get_std_opt
from utils.optimization.ReduceLROnPlateau_def import ReduceLROnPlateau
from utils.optimization import build_optimizer
from model.Transformer_def import Transformer


def train(
    model_id,
    sequences_per_img=5,
    batch_size=10,
    resnet_conv_feature_size=2048,
    start_from=None,
    input_json_file_name=None,
    input_label_h5_file_name=None,
    label_smoothing=0,
    structure_loss_weight=1,
    train_sample_method="sample",
    train_beam_size=1,
    struc_use_logsoftmax=True,
    train_sample_n=5,
    structure_loss_type="seqnll",
    optimizer_type=NOAM,
    noamopt_factor=1,
    noamopt_warmup=20000,
    core_optimizer="sgd",
    learning_rate=0.0005,
    optimizer_alpha=0.9,
    optimizer_beta=0.999,
    optimizer_epsilon=1e-8,
    weight_decay=0,
    load_best_score=True,
    max_epochs=50,
    scheduled_sampling_start=-1,
    scheduled_sampling_increase_every=5,
    scheduled_sampling_increase_prob=0.05,
    scheduled_sampling_max_prob=0.25,
    self_critical_after=-1,
    structure_after=-1,
    cached_tokens="coco-train-idxs",
    grad_clip_value=0.1,
    grad_clip_mode=CLIP_VALUE,
    log_loss_iterations=25,
    save_every_epoch=True,
    save_checkpoint_iterations=3000,
    save_history_ckpt=True,
    eval_language_model=True,
):

    #
    # File names
    info_file_name = (
        join(start_from, "infos_" + model_id + ".pkl") if start_from is not None else ""
    )
    history_file_name = (
        join(start_from, "histories_" + model_id + ".pkl")
        if start_from is not None
        else ""
    )
    model_file_name = join(start_from, "model.pth") if start_from is not None else ""
    optimizer_file_name = (
        join(start_from, "optimizer.pth") if start_from is not None else ""
    )

    #
    # Load data
    loader = DataLoader(
        sequences_per_img,
        batch_size=batch_size,
        use_fc=True,
        use_att=True,
        use_box=0,
        norm_att_feat=0,
        norm_box_feat=0,
        input_json_file_name=input_json_file_name,
        input_label_h5_file_name=input_label_h5_file_name,
    )
    vocab_size = loader.vocab_size
    seq_length = loader.seq_length

    #
    # Initialize training info
    infos = {
        "iter": 0,
        "epoch": 0,
        "loader_state_dict": None,
        "vocab": loader.get_vocab(),
    }

    #
    # Load existing state training information, if there is any
    if start_from is not None and isfile(info_file_name):
        #
        with open(info_file_name, "rb") as f:
            assert True

    #
    # Create data logger
    histories = defaultdict(dict)
    if start_from is not None and isfile(history_file_name):
        with open(history_file_name, "rb") as f:
            histories.update(pickle_load(f))

    # tensorboard logger
    tb_summary_writer = SummaryWriter(checkpoint_path)

    #
    # Create our model
    vocab = loader.get_vocab()
    model = Transformer(
        vocab_size, resnet_conv_feature_size=resnet_conv_feature_size
    ).cuda()

    #
    # Load pretrained weights:
    if start_from is not None and isfile(model_file_name):
        model.load_state_dict(torch_load(model_file_name))

    #
    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    lw_model = LossWrapper(
        model,
        label_smoothing=label_smoothing,
        structure_loss_weight=structure_loss_weight,
        train_sample_method=train_sample_method,
        train_beam_size=train_beam_size,
        struc_use_logsoftmax=struc_use_logsoftmax,
        train_sample_n=train_sample_n,
        structure_loss_type=structure_loss_type,
    )

    #
    # Wrap with dataparallel
    dp_model = DataParallel(model)
    dp_lw_model = DataParallel(lw_model)

    #
    #  Build optimizer
    if optimizer_type == NOAM:
        optimizer = get_std_opt(model, factor=noamopt_factor, warmup=noamopt_warmup)
    elif optimizer_type == REDUCE_LR:
        optimizer = build_optimizer(
            model.parameters(),
            core_optimizer=core_optimizer,
            learning_rate=learning_rate,
            optimizer_alpha=optimizer_alpha,
            optimizer_beta=optimizer_beta,
            optimizer_epsilon=optimizer_epsilon,
            weight_decay=weight_decay,
        )
        optimizer = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        raise (
            Exception("Only supports NoamOpt and ReduceLROnPlateau optimization types")
        )

    #
    # # Load the optimizer
    if start_from is not None and isfile(optimizer_file_name):
        optimizer.load_state_dict(torch_load(optimizer_file_name))

    #
    # Prepare for training
    iteration = infos["iter"]
    epoch = infos["epoch"]
    #
    # For back compatibility
    if "iterators" in infos:
        infos["loader_state_dict"] = {
            split: {
                "index_list": infos["split_ix"][split],
                "iter_counter": infos["iterators"][split],
            }
            for split in ["train", "val", "test"]
        }
    loader.load_state_dict(infos["loader_state_dict"])
    if load_best_score == 1:
        best_val_score = infos.get("best_val_score", None)
    if optimizer_type == NOAM:
        optimizer._step = iteration
    #
    # Assure in training mode
    dp_lw_model.train()
    epoch_done = True

    #
    # Start training
    try:
        while True:
            #
            # Check max epochs
            if epoch >= max_epochs and max_epochs != -1:
                break

            #
            # Update end of epoch data
            if epoch_done:
                #
                # Assign the scheduled sampling prob
                if epoch > scheduled_sampling_start and scheduled_sampling_start >= 0:
                    frac = (
                        epoch - scheduled_sampling_start
                    ) // scheduled_sampling_increase_every
                    ss_prob = min(
                        scheduled_sampling_increase_prob * frac,
                        scheduled_sampling_max_prob,
                    )
                    model.ss_prob = ss_prob

                #
                # If start self critical training
                if self_critical_after != -1 and epoch >= self_critical_after:
                    sc_flag = True
                    init_scorer(cached_tokens)
                else:
                    sc_flag = False

                #
                # If start structure loss training
                if structure_after != -1 and epoch >= structure_after:
                    struc_flag = True
                    init_scorer(cached_tokens)
                else:
                    struc_flag = False

                #
                # End epoch update
                epoch_done = False
            #
            # Compute time to load data
            start = time.time()
            data = loader.get_batch("train")
            load_data_time = time.time() - start
            print(f"Time to load data: {load_data_time} seconds")

            ########################
            # SYNC
            ########################
            synchronize()

            #
            # Compute time to complete epoch
            start = time.time()

            #
            # Make sure data is in GPU memory
            tmp = [
                data["fc_feats"],
                data["att_feats"],
                data["labels"],
                data["masks"],
                data["att_masks"],
            ]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            #
            # Reset gradient
            optimizer.zero_grad()

            #
            print("MADE IT TO THE MODEL EVALUATION")
            #
            # Evaluate model
            model_out = dp_lw_model(
                fc_feats,
                att_feats,
                labels,
                masks,
                att_masks,
                data["gts"],
                torch_arange(0, len(data["gts"])),
                sc_flag,
                struc_flag,
            )

            #
            # Average loss over training batch
            loss = model_out["loss"].mean()

            #
            # Compute gradient
            loss.backward()

            #
            # Clip gradient
            if grad_clip_value != 0:
                gradient_clipping_functions[grad_clip_mode](
                    model.parameters(), grad_clip_value
                )
            #
            # Update
            optimizer.step()
            train_loss = loss.item()
            end = time.time()

            ########################
            # SYNC
            ########################
            synchronize()

            #
            # Output status
            if struc_flag:
                print(
                    "iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}".format(
                        iteration,
                        epoch,
                        train_loss,
                        model_out["lm_loss"].mean().item(),
                        model_out["struc_loss"].mean().item(),
                        end - start,
                    )
                )
            elif not sc_flag:
                print(
                    "iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                        iteration, epoch, train_loss, end - start
                    )
                )
            else:
                print(
                    "iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}".format(
                        iteration, epoch, model_out["reward"].mean(), end - start
                    )
                )

            #
            # Update the iteration and epoch
            iteration += 1
            if data["bounds"]["wrapped"]:
                epoch += 1
                epoch_done = True

            #
            # Write the training loss summary
            if iteration % log_loss_iterations == 0:

                tb_summary_writer.add_scalar("train_loss", train_loss, iteration)

                if optimizer_type == NOAM:
                    current_lr = optimizer.rate()
                elif optimizer_type == REDUCE_LR:
                    current_lr = optimizer.current_lr

                tb_summary_writer.add_scalar("learning_rate", current_lr, iteration)
                tb_summary_writer.add_scalar(
                    "scheduled_sampling_prob", model.ss_prob, iteration
                )

                if sc_flag:
                    tb_summary_writer.add_scalar(
                        "avg_reward", model_out["reward"].mean(), iteration
                    )
                elif struc_flag:
                    tb_summary_writer.add_scalar(
                        "lm_loss", model_out["lm_loss"].mean().item(), iteration
                    )
                    tb_summary_writer.add_scalar(
                        "struc_loss", model_out["struc_loss"].mean().item(), iteration
                    )
                    tb_summary_writer.add_scalar(
                        "reward", model_out["reward"].mean().item(), iteration
                    )
                    tb_summary_writer.add_scalar(
                        "reward_var", model_out["reward"].var(1).mean(), iteration
                    )

                histories["loss_history"][iteration] = (
                    train_loss if not sc_flag else model_out["reward"].mean()
                )
                histories["lr_history"][iteration] = current_lr
                histories["ss_prob_history"][iteration] = model.ss_prob

            #
            # Update infos
            infos["iter"] = iteration
            infos["epoch"] = epoch
            infos["loader_state_dict"] = loader.state_dict()

            #
            # Make evaluation on validation set, and save model
            if (
                iteration % save_checkpoint_iterations == 0 and not save_every_epoch
            ) or (epoch_done and save_every_epoch):
                #
                # Evaluate model on Validation set of COCO
                eval_kwargs = {"split": "val", "dataset": input_json_file_name}
                val_loss, predictions, lang_stats = eval_split(
                    dp_model,
                    lw_model.crit,
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
                )

                #
                # Reduces learning rate if no improvement in objective
                if optimizer_type == REDUCE_LR:
                    if "CIDEr" in lang_stats:
                        optimizer.scheduler_step(-lang_stats["CIDEr"])
                    else:
                        optimizer.scheduler_step(val_loss)

                #
                # Write validation result into summary
                tb_summary_writer.add_scalar("validation loss", val_loss, iteration)
                if lang_stats is not None:
                    for k, v in lang_stats.items():
                        tb_summary_writer.add_scalar(k, v, iteration)

                histories["val_result_history"][iteration] = {
                    "loss": val_loss,
                    "lang_stats": lang_stats,
                    "predictions": predictions,
                }

                #
                # Save model if is improving on validation result
                if eval_language_model:
                    current_score = lang_stats["CIDEr"]
                else:
                    current_score = -val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                #
                # Dump miscalleous informations
                infos["best_val_score"] = best_val_score

                #
                # Save checkpoints...seems only most recent one keep histories,
                # and it's overwritten each time
                save_checkpoint(
                    model,
                    infos,
                    optimizer,
                    checkpoint_dir=checkpoint_path,
                    histories=histories,
                    append="RECENT",
                )
                if save_history_ckpt:
                    save_checkpoint(
                        model,
                        infos,
                        optimizer,
                        checkpoint_dir=checkpoint_path,
                        append=str(epoch) if save_every_epoch else str(iteration),
                    )
                if best_flag:
                    save_checkpoint(
                        model,
                        infos,
                        optimizer,
                        checkpoint_dir=checkpoint_path,
                        append="BEST",
                    )

    except (RuntimeError, KeyboardInterrupt):
        print(f'{BAR("=", 20)}Save checkpoint on exception...')
        save_checkpoint(
            model, infos, optimizer, checkpoint_dir=checkpoint_path, append="EXCEPTION"
        )
        print(f'...checkpoint saved.{BAR("=", 20)}')
        stack_trace = format_exc()
        print(stack_trace)


#
# Load default values
cn = CfgNode(CfgNode.load_yaml_with_base("defaults.yml"))

# for k,v in cn.items():
#     print(f"{k}: {v}")

# Check if args are valid
# assert args.rnn_size > 0, "rnn_size should be greater than 0"
# assert args.num_layers > 0, "num_layers should be greater than 0"
# assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
# assert args.batch_size > 0, "batch_size should be greater than 0"
# assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
# assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
# assert args.beam_size > 0, "beam_size should be greater than 0"
# assert args.save_checkpoint_iterations > 0, "save_checkpoint_iterations should be greater than 0"
# assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
# assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
# assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
# assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

train(
    "FAKE_MODEL",
    batch_size=10,
    resnet_conv_feature_size=cn.resnet_conv_feature_size,
    sequences_per_img=cn.sequences_per_img,
    input_json_file_name=cn.input_json_file_name,
    input_label_h5_file_name=cn.input_label_h5_file_name,
    start_from=None,
    label_smoothing=0,
    structure_loss_weight=1,
    train_sample_method="sample",
    train_beam_size=1,
    struc_use_logsoftmax=True,
    train_sample_n=5,
    structure_loss_type="seqnll",
    optimizer_type=NOAM,
    noamopt_factor=1,
    noamopt_warmup=20000,
    core_optimizer="sgd",
    learning_rate=0.0005,
    optimizer_alpha=0.9,
    optimizer_beta=0.999,
    optimizer_epsilon=1e-8,
    weight_decay=0,
    load_best_score=True,
    max_epochs=50,
    scheduled_sampling_start=-1,
    scheduled_sampling_increase_every=5,
    scheduled_sampling_increase_prob=0.05,
    scheduled_sampling_max_prob=0.25,
    self_critical_after=-1,
    structure_after=-1,
    cached_tokens="coco-train-idxs",
    grad_clip_value=0.1,
    grad_clip_mode=CLIP_VALUE,
    log_loss_iterations=25,
    save_every_epoch=True,
    save_checkpoint_iterations=3000,
    eval_language_model=True,
)
