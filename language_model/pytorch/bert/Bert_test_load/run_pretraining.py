# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 MLBenchmark Group. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT Pretraining"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import h5py
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import logging
import math
import multiprocessing
import numpy as np
import os
import random
import re
import time

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from modeling import BertForPreTraining, BertConfig
from schedulers import LinearWarmupPolyDecayScheduler

#import utils
import utils_bert as utils  #TUAN
try:
    import apex.amp as amp
except:
    print ("[TUAN] Not using APEX")

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

# from apex.optimizers import FusedLAMB
#from lamb import Lamb


try:
    moreh_ops = torch.ops.moreh
    Lamb = moreh_ops.FusedLAMB
    moreh_enable = True
    print("Use Moreh's FusedLAMB")
except:
    #from lamb import Lamb
    #print("Use Bert's LAMB")
    from apex.optimizers import FusedLAMB as Lamb
    moreh_enable = False
    print("Use Apex's FusedLAMB")

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForPreTraining, BertConfig
from schedulers import LinearWarmUpScheduler, LinearWarmupPolyDecayScheduler

#TUAN
torch.multiprocessing.set_sharing_strategy('file_system')

#TUAN
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.manual_seed(0)
#torch.cuda.manual_seed_all(0)
#np.random.seed(0)
#random.seed(0)

# Global variables
skipped_steps = 0
cached_batches = []

class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args, worker_init_fn):
    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
        batch_size=args.train_batch_size, num_workers=0, worker_init_fn=worker_init_fn,
        pin_memory=True)

    return train_dataloader, input_file

def create_eval_dataset(args, worker_init_fn):
    eval_data = []
    for eval_file in sorted(os.listdir(args.eval_dir)):
        eval_file_path = os.path.join(args.eval_dir, eval_file)
        if os.path.isfile(eval_file_path) and 'part' in eval_file_path:
            eval_data.extend(pretraining_dataset(eval_file_path, max_pred_length=args.max_predictions_per_seq))
            if len(eval_data) > args.num_eval_examples:
                eval_data = eval_data[:args.num_eval_examples]
                break
    if torch.distributed.is_initialized():
        chunk_size = args.num_eval_examples // torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        remainder = args.num_eval_examples % torch.distributed.get_world_size()
        if rank<remainder:
            eval_data = eval_data[(chunk_size+1)*rank : (chunk_size+1)*(rank+1)]
        else:
            eval_data = eval_data[chunk_size*rank+remainder : chunk_size*(rank+1)+remainder]


    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                         num_workers=0, worker_init_fn=worker_init_fn, pin_memory=True)

    return eval_dataloader

class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]

def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        help="The eval data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--eval_iter_start_samples",
                        default=3000000,
                        type=int,
                        help="Sample to begin performing eval.")
    parser.add_argument("--eval_iter_samples",
                        default=-1,
                        type=int,
                        help="If set to -1, disable eval, \
                        else evaluate every eval_iter_samples during training")
    parser.add_argument("--num_eval_examples",
                        default=10000,
                        type=int,
                        help="number of eval examples to run eval on")
    parser.add_argument("--cache_eval_data",
                        default=False,
                        action='store_true',
                        help="whether to cache evaluation data on GPU")

    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")
    parser.add_argument("--init_tf_checkpoint",
                        default=None,
                        type=str,
                        help="The initial TF checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=76,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=18,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=4e-5,
                        type=float,
                        help="The initial learning rate for LAMB.")
    parser.add_argument("--weight_decay_rate",
                        default=0.01,
                        type=float,
                        help="weight decay rate for LAMB.")
    parser.add_argument("--opt_lamb_beta_1",
                        default=0.9,
                        type=float,
                        help="LAMB beta1.")
    parser.add_argument("--opt_lamb_beta_2",
                        default=0.999,
                        type=float,
                        help="LAMB beta2.")
    parser.add_argument("--max_steps",
                        default=1536,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--max_samples_termination",
                        default=14000000,
                        type=float,
                        help="Total number of training samples to run.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of optimizer update steps to perform linear learning rate warmup for. "
                             "Typically 1/8th of steps for Phase2")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=float,
                        help="Number of optimizer update steps to perform linear learning rate warmup for. "
                             "Typically 1/8th of steps for Phase2")
    parser.add_argument("--start_warmup_step",
                        default=0,
                        type=float,
                        help="Starting step for warmup. ")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint. If set, precedes init_checkpoint/init_tf_checkpoint")
    parser.add_argument('--keep_n_most_recent_checkpoints',
                        type=int,
                        default=20,
                        help="Number of checkpoints to keep (rolling basis).")
    parser.add_argument('--num_samples_per_checkpoint',
                        type=int,
                        default=500000,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--min_samples_to_start_checkpoints',
                        type=int,
                        default=3000000,
                        help="Number of update steps until model checkpoints start saving to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Only required for checkpoint saving format")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--unpad",
                        default=False,
                        action='store_true',
                        help="Whether to run with unpadding.")
    parser.add_argument("--pad",
                        default=False,
                        action='store_true',
                        help="Whether to pad tokens.")
    parser.add_argument("--enable_fuse_dropout",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of attention mask to softmax and dropout.")
    parser.add_argument("--disable_fuse_mask",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the attention mask to softmax.")
    parser.add_argument("--disable_fuse_scale",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the scaling to BMM1.")
    parser.add_argument("--disable_fuse_qkv",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the QKV GEMMs.")
    parser.add_argument("--disable_apex_softmax",
                        default=False,
                        action='store_true',
                        help="Whether to disable apex softmax.")
    parser.add_argument("--enable_stream",
                        default=False,
                        action='store_true',
                        help="Enable use of streams for pad case.")
    parser.add_argument("--fused_gelu_bias",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--dense_seq_output",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--bert_config_path',
                        type=str,
                        default="/workspace/phase1",
                        help="Path bert_config.json is located in")
    parser.add_argument('--target_mlm_accuracy',
                        type=float,
                        default=0.0,
                        help="Stop training after reaching this Masked-LM accuracy")
    parser.add_argument('--train_mlm_accuracy_window_size',
                        type=int,
                        default=0,
                        help="Average accuracy over this amount of batches before performing a stopping criterion test")
    parser.add_argument('--num_epochs_to_generate_seeds_for',
                        type=int,
                        default=2,
                        help="Number of epochs to plan seeds for. Same set across all workers.")
    #TUAN
    parser.add_argument('--opt',
                        type=str,
                        default="O2",
                        help="Apex amp opt level")

    args = parser.parse_args()

    # Check we've been given a checkpoint
    assert args.init_checkpoint is not None or args.init_tf_checkpoint is not None or found_resume_checkpoint(args), \
        "Must specify --init_checkpoint, --init_tf_checkpoint or have ckpt to resume from in --output_dir of the form *.pt"

    assert not (args.init_checkpoint is not None and args.init_tf_checkpoint is not None), \
            "Can only specify one of --init_checkpoint and --init_tf_checkpoint"

    return args

# Returns true only if resuming from a checkpoint found in output_dir. 
# init_checkpoint and init_tf_checkpoint are not considered
def found_resume_checkpoint(args):
    if args.phase2:
        checkpoint_str = "phase2_ckpt*.pt"
    else:
        checkpoint_str = "phase1_ckpt*.pt"
    return args.resume_from_checkpoint and len(glob.glob(os.path.join(args.output_dir, checkpoint_str))) > 0

def setup_training(args):
    #assert (torch.cuda.is_available())

    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1 #TUAN 
        print ("n_gpu :", args.n_gpu)
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.n_gpu = torch.distributed.get_world_size()

    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not (args.do_train or (args.eval_dir and args.eval_iter_samples <= 0)):
        raise ValueError(" `do_train`  or should be in offline eval mode")

    if not args.resume_from_checkpoint or not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def remap_attn_parameters(model_dict):
    res_dict = OrderedDict()
    for k in model_dict:
        if 'attention' in k:
            if 'self.query.weight' in k:
                new_k = k.replace('self.query.weight', 'multi_head_attention.q_weight')
            elif 'self.key.weight' in k:
                new_k = k.replace('self.key.weight', 'multi_head_attention.k_weight')
            elif 'self.value.weight' in k:
                new_k = k.replace('self.value.weight', 'multi_head_attention.v_weight')
            elif 'self.query.bias' in k:
                new_k = k.replace('self.query.bias', 'multi_head_attention.q_bias')
            elif 'self.key.bias' in k:
                new_k = k.replace('self.key.bias', 'multi_head_attention.k_bias')
            elif 'self.value.bias' in k:
                new_k = k.replace('self.value.bias', 'multi_head_attention.v_bias')
            elif 'output.dense.weight' in k:
                new_k = k.replace('output.dense.weight', 'multi_head_attention.out_proj_weight')
            elif 'output.dense.bias' in k:
                new_k = k.replace('output.dense.bias', 'multi_head_attention.out_proj_bias')
            elif 'output.LayerNorm.weight' in k:
                new_k = k.replace('output.LayerNorm.weight', 'layer_norm.weight')
            elif 'output.LayerNorm.bias' in k:
                new_k = k.replace('output.LayerNorm.bias', 'layer_norm.bias')
            else:
                new_k = k
        else:
            new_k = k
        res_dict[new_k] = model_dict[k]
    model_dict.clear()
    return res_dict

def prepare_model_and_optimizer(args, device):
    global_step = 0
    args.resume_step = 0
    checkpoint = None

    # FIXME
    #assert args.fp16 == False and "[Moreh/SnuDL] FP16 training not supported"
    #assert args.local_rank == -1 and "[Moreh/SnuDL] Distributed training not supported"

    config = BertConfig.from_json_file(args.bert_config_path)
    config.fused_gelu_bias = args.fused_gelu_bias
    config.dense_seq_output = args.dense_seq_output
    config.unpad = args.unpad
    config.pad = args.pad
    config.fuse_qkv = not args.disable_fuse_qkv
    config.fuse_scale = not args.disable_fuse_scale
    config.fuse_mask = not args.disable_fuse_mask
    config.fuse_dropout = args.enable_fuse_dropout
    config.apex_softmax = not args.disable_apex_softmax
    config.enable_stream = args.enable_stream
    if config.fuse_mask == True: config.apex_softmax = True
    if config.pad == False: config.enable_stream = True



    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    # Load from Pyt checkpoint - either given as init_checkpoint, or picked up from output_dir if found
    if args.init_checkpoint is not None or found_resume_checkpoint(args):
        # Prepare model

        model = BertForPreTraining(config)
        if args.init_checkpoint is None: # finding checkpoint in output_dir
            checkpoint_str = "phase2_ckpt_*.pt" if args.phase2 else "phase1_ckpt_*.pt"
            model_names = [f for f in glob.glob(os.path.join(args.output_dir, checkpoint_str))]
            global_step = max([int(x.split('.pt')[0].split('_')[-1].strip()) for x in model_names])
            args.resume_step = global_step #used for throughput computation

            resume_init_checkpoint = os.path.join(args.output_dir, checkpoint_str.replace("*", str(global_step)))
            print("Setting init checkpoint to %s - which is the latest in %s" %(resume_init_checkpoint, args.output_dir))
            checkpoint=torch.load(resume_init_checkpoint, map_location="cpu")
        else:
            checkpoint=torch.load(args.init_checkpoint, map_location="cpu")["model"]

        model.load_state_dict(checkpoint, strict=True)

    else: #Load from TF Checkpoint
        model = BertForPreTraining.from_pretrained(args.init_tf_checkpoint, from_tf=True, config=config)


    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # FIXME
    optimizer = Lamb(optimizer_grouped_parameters,
                     lr=args.learning_rate,
                     betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2))
    # optimizer = FusedLAMB(optimizer_grouped_parameters,
    #                       lr=args.learning_rate,
    #                       betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2))
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters,
    #                       lr=args.learning_rate,
    #                       betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2))
    b1, b2 = optimizer.defaults['betas']

    if args.warmup_steps == 0:
        warmup_steps = int(args.max_steps * args.warmup_proportion)
        warmup_start = 0
    else:
        warmup_steps = args.warmup_steps
        warmup_start = args.start_warmup_step
    lr_scheduler = LinearWarmupPolyDecayScheduler(optimizer, start_warmup_steps=warmup_start, warmup_steps=warmup_steps,
                                                  total_steps=args.max_steps, end_learning_rate=0.0, degree=1.0)

    if found_resume_checkpoint(args):
        optimizer.load_state_dict(checkpoint['optimizer']) #restores m,v states (only if resuming checkpoint, not for init_checkpoint and init_tf_checkpoint for now)


    #TUAN
    if args.fp16:
        print ("[TUAN] Using amp FP16")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt, max_loss_scale=512.0)



    return model, optimizer, lr_scheduler, checkpoint, global_step

def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):

    global skipped_steps
    optimizer.step()
    for param in model.parameters():
        param.grad = None
    global_step += 1

    return global_step

def run_eval(model, eval_dataloader, device, num_eval_examples, first_eval=False, use_cache=False):
    print ("RUNNING EVAL...")
    model.eval()

    total_eval_loss, total_eval_mlm_acc = 0.0, 0.0
    total_masked = 0

    # on first eval, load and cache data on GPU
    if first_eval and use_cache:
        for batch in eval_dataloader:
            cached_batches.append([t.to(device) for t in batch])

    with torch.no_grad():
        i = 0
        for batch in cached_batches if use_cache else eval_dataloader:
            if not use_cache: batch = [t.to(device) for t in batch]
            input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
            force_save = (i == 6)
            loss, mlm_acc, num_masked = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                  masked_lm_labels=masked_lm_labels,
                                  next_sentence_label=next_sentence_labels,
                                  force_save=force_save)
            total_eval_loss += loss * num_masked
            total_eval_mlm_acc += mlm_acc * num_masked
            total_masked += num_masked
            i += 1
    model.train()

    #total_eval_mlm_acc and total_eval_loss are already tensors, total_masked is not 
    total_masked = torch.tensor(total_masked, device=device, dtype=torch.int64)

    if torch.distributed.is_initialized():
        #Collect total scores from all ranks
        torch.distributed.all_reduce(total_eval_mlm_acc, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_eval_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_masked, op=torch.distributed.ReduceOp.SUM)

    # Average by number of examples
    total_eval_mlm_acc /= total_masked
    total_eval_loss /= total_masked

    return total_eval_loss.item(), total_eval_mlm_acc.item()


def main():
    args = parse_arguments()
    status = 'aborted'  # later set to 'success' if termination criteria met

    if args.use_env and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    device, args = setup_training(args)

    print ("TUAN:global_batch_size: ", global_batch_size(args))
    worker_seeds, shuffling_seeds = utils.setup_seeds(args.seed, args.num_epochs_to_generate_seeds_for, device)
    worker_seed = worker_seeds[0] # FIXME
    # worker_seed = worker_seeds[torch.distributed.get_rank()]
        
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitObj(worker_seed)

    if utils.is_main_process():
        print("parsed args:")
        print(args)
    # Prepare optimizer
    model, optimizer, lr_scheduler, checkpoint, global_step = prepare_model_and_optimizer(args, device)
    samples_trained = global_step * args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu

    final_loss = float("inf")
    train_time_raw = float("inf")
    raw_train_start = time.time()

    #print (model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print ("#params1: ", pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print ("#params2: ", pytorch_total_params)
    print ("optimizer: ", optimizer)
  
    _sum = 0
    for p in model.parameters():
        _sum += p.sum()
    print ("model sum : ", _sum)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print ("name: ", name)
            print ("sum: ", param.sum())
    


    if args.do_train:
        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 1
        training_steps = 0
        end_training, converged = False, False
        samples_trained_prev = 0
        eval_count = 0

        pool = ProcessPoolExecutor(1)

        if args.target_mlm_accuracy:
            if args.train_mlm_accuracy_window_size > 0:
                accuracy_scores = []
                avg_mlm_accuracy = torch.Tensor([0]).cuda()
        

        first_epoch = True
        if found_resume_checkpoint(args):
            f_start_id = checkpoint['files'][0]
            files = checkpoint['files'][1:]
            num_files = len(files)
        else:
            files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                     os.path.isfile(os.path.join(args.input_dir, f)) and 'part' in f]
            files.sort()
            num_files = len(files)
            random.Random(shuffling_seeds[epoch]).shuffle(files)
            f_start_id = 0


        # Start prefetching eval dataset
        if args.eval_dir:
            # KDH : ProcessExecutor로 실행하면 shared_memory_object를 rw 모드로 읽을 수 없는 이슈가 있음. moreh뿐만 아니라 origin torch에서도. 그래서 메인 스레드에서 실행하는걸로 수정.
            #eval_dataset_future = pool.submit(create_eval_dataset, args, worker_init_fn=worker_init)
            eval_dataset_future = create_eval_dataset

        while global_step < args.max_steps and not end_training:

            if utils.is_main_process():
                print("parsed args:")
                print(args)

                now_time = time.time()
                now_step = global_step
                now_skipped = skipped_steps

                print("epoch:", epoch)

            thread = None

            # Reshuffle file list on subsequent epochs
            if not first_epoch:
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f)) and 'part' in f]
                files.sort()
                num_files = len(files)
                random.Random(shuffling_seeds[epoch]).shuffle(files)
                f_start_id = 0

            first_epoch = False

            shared_file_list = {}

            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
                remainder = torch.distributed.get_world_size() % num_files
                data_file = files[(f_start_id*torch.distributed.get_world_size() + torch.distributed.get_rank() +
                                   remainder * f_start_id) % num_files]
            else:
                # data_file = files[(f_start_id*torch.distributed.get_world_size() + torch.distributed.get_rank()) % num_files]
                data_file = files[f_start_id % num_files] # FIXME

            previous_file = data_file

            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                          batch_size=args.train_batch_size, num_workers=0, worker_init_fn=worker_init, pin_memory=True)

            print ("args.train_batch_size: ", args.train_batch_size)
            overflow_buf = None
            if args.allreduce_post_accumulation:
                overflow_buf = torch.cuda.IntTensor([0])

            for f_id in range(f_start_id + 1, len(files)):
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
                    data_file = files[(f_id*torch.distributed.get_world_size() + torch.distributed.get_rank() +
                                       remainder * f_id) % num_files]
                else:
                    # data_file = files[(f_id*torch.distributed.get_world_size() + torch.distributed.get_rank())%num_files]
                    data_file = files[f_id % num_files] # FIXME

                previous_file = data_file

                dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args, worker_init_fn=worker_init)

                print ("[TUAN]: len(train_dataloader): ", len(train_dataloader))
                start_time = time.time()
                time_point = time.time()
                for step, batch in enumerate(train_dataloader):
                    training_steps += 1
                    update_step = training_steps % args.gradient_accumulation_steps == 0

                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    loss, mlm_acc, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels,
                                    checkpoint_activations=args.checkpoint_activations)

                    divisor = args.gradient_accumulation_steps
                    if args.gradient_accumulation_steps > 1:
                        if not args.allreduce_post_accumulation:
                            # this division was merged into predivision
                            loss = loss / args.gradient_accumulation_steps
                            divisor = 1.0

                    #loss.backward()
                    #TUAN: Changed to scaled_loss:
                    if step > 96 and False:
                        print ("LOSS: ", loss.item())
                        import sys
                        sys.exit(0)
                    
                    print ("LOSS: ", loss.item())
                    
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    average_loss += loss.item()

                    ###TUAN: Debugging the Nan issues
                    if False:
                        print ("Examining the model weight gradients...")
                        for name, param in model.named_parameters():
                            if param.requires_grad == True:
                                grad = param.grad                        
                                if grad is not None:
                                    tmp = torch.isnan(grad)
                                    if torch.sum(tmp).item() > 0:
                                        print ("Nan: ", torch.sum(tmp))
                                        print ("param: ", name)
                        print ("DONE")
                        import sys
                        sys.exit(0)

                    #if update_step:
                    if True:
                        lr_scheduler.step()  # learning rate warmup
                        #global_step = take_optimizer_step(args, optimizer, model, overflow_buf, global_step)
                        global_step = 1
                        samples_trained = global_step * args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu

                        #if (args.eval_dir and args.eval_iter_samples > 0 and
                        #    samples_trained >= args.eval_iter_start_samples + eval_count * args.eval_iter_samples):
                        if True:

                            # on first eval, get eval_dataloader
                            if eval_count == 0: 
                               #eval_dataloader = eval_dataset_future.result(timeout=None)
                               eval_dataloader = eval_dataset_future(args,worker_init)

                            samples_trained_prev = samples_trained
                            eval_avg_loss, eval_avg_mlm_accuracy = run_eval(model, eval_dataloader, device, args.num_eval_examples,
                                                                            first_eval=(eval_count == 0), use_cache=args.cache_eval_data)
                            if utils.is_main_process():
                                print({"global_steps": global_step, "eval_loss": eval_avg_loss, "eval_mlm_accuracy":eval_avg_mlm_accuracy})

                            if args.target_mlm_accuracy:
                                if eval_avg_mlm_accuracy >= args.target_mlm_accuracy:
                                    end_training, converged = True, True
                                    if utils.is_main_process():
                                        print("%f > %f, Target MLM Accuracy reached at %d"%(eval_avg_mlm_accuracy, args.target_mlm_accuracy, global_step))

                            eval_count += 1
                            import sys
                            sys.exit(0)

                    if args.target_mlm_accuracy and args.train_mlm_accuracy_window_size > 0:
                        accuracy_scores.append(mlm_acc)
                        if update_step:
                            accuracy_scores = accuracy_scores[-args.train_mlm_accuracy_window_size * args.gradient_accumulation_steps:]
                            avg_mlm_accuracy[0] = sum(accuracy_scores) / len(accuracy_scores)
                            torch.distributed.all_reduce(avg_mlm_accuracy, op=torch.distributed.ReduceOp.SUM)
                            avg_mlm_accuracy /= torch.distributed.get_world_size()

                    if training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        samples_trained = global_step * args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu
                        if utils.is_main_process():
                            time_interval = time.time() - now_time
                            step_interval = global_step - now_step
                            skip_interval = skipped_steps - now_skipped
                            now_time = time.time()
                            now_step = global_step
                            now_skipped = skipped_steps
                            training_perf = args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu \
                                            * (step_interval + skip_interval) / time_interval
                           

                            start_time = time.time()
                            if args.train_mlm_accuracy_window_size > 0:
                                print(("training_steps: {}, step_loss: {:.3f}, learning_rate: {:.2E}, " +
                                       "seq/s: {:.0f}, global_steps: {}, samples_trained: {}, elapsed_time: {}, mlm_accuracy: {:.2E}").format(
                                       training_steps,  
                                       loss.item() * args.gradient_accumulation_steps / divisor,
                                       optimizer.param_groups[0]['lr'], training_perf, now_step,
                                       samples_trained, avg_mlm_accuracy[0].item(), time.time() - time_point
                                       ))
                            else:
                                print(("training_steps: {}, step_loss: {:.3f}, learning_rate: {:.2E}, " +
                                       "seq/s: {:.0f}, global_steps: {}, samples_trained: {}, elapsed_time: {}").format(
                                       training_steps, loss.item() * args.gradient_accumulation_steps / divisor,
                                       optimizer.param_groups[0]['lr'], training_perf, now_step,
                                       samples_trained, time.time() - time_point
                                       ))
                            time_point = time.time()

                        average_loss = 0

                    if global_step >= args.max_steps or end_training:
                        status = 'success' if converged else 'aborted'
                        end_training = True
                        train_time_raw = time.time() - raw_train_start
                        last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / (last_num_steps * divisor)
                        if (torch.distributed.is_initialized()):
                            average_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        final_loss = average_loss.item()
                        if utils.is_main_process():
                            if args.train_mlm_accuracy_window_size > 0:
                                print((epoch, training_steps / args.gradient_accumulation_steps, ), {"final_loss": final_loss,
                                    "final_mlm_accuracy": avg_mlm_accuracy[0].item()})
                            else:
                                print((epoch, training_steps / args.gradient_accumulation_steps, ), {"final_loss": final_loss})

                    if end_training or (samples_trained - samples_trained_prev >= args.num_samples_per_checkpoint and samples_trained >= args.min_samples_to_start_checkpoints):
                        samples_trained_prev = samples_trained
                        if utils.is_main_process() and not args.skip_checkpoint:
                            # Save a trained model
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            if args.phase2:
                                output_save_file = os.path.join(args.output_dir, "phase2_ckpt_{}.pt".format(samples_trained))
                            else:
                                output_save_file = os.path.join(args.output_dir, "phase1_ckpt_{}.pt".format(samples_trained))
                            if args.do_train:
                                torch.save({'model': model_to_save.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            # 'master params': list(amp.master_params(optimizer)),
                                            'files': [f_id] + files}, output_save_file)

                                most_recent_ckpts_paths.append(output_save_file)
                                if len(most_recent_ckpts_paths) > args.keep_n_most_recent_checkpoints:
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed)
                            
                        if samples_trained >= args.max_samples_termination or end_training:
                            status = 'success' if converged else 'aborted' 
                            end_training = True
                            break

                del train_dataloader

                if samples_trained >= args.max_samples_termination or end_training:
                    status = 'success' if converged else 'aborted'
                    end_training = True
                    break

                train_dataloader, data_file = dataset_future.result(timeout=None)

            epoch += 1

    print("Training result: ", status)

    return args, final_loss, train_time_raw

def global_batch_size(args):
    return args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu

if __name__ == "__main__":

    now = time.time()
    args, final_loss, train_time_raw = main()

    gpu_count = args.n_gpu
    if torch.distributed.is_initialized():
        gpu_count = torch.distributed.get_world_size()
    if utils.is_main_process():
        e2e_time = time.time() - now
        training_perf = global_batch_size(args) \
                        * (args.max_steps - args.resume_step + skipped_steps) / train_time_raw
        if args.do_train:
            print({"e2e_time": e2e_time, "training_sequences_per_second": training_perf,
                                             "final_loss": final_loss, "raw_train_time": train_time_raw })
        else:
            print({"e2e_time": e2e_time})
