from numpy import NaN
import torch
import os
import argparse
from transformers import Adafactor
from tqdm import tqdm

import pandas as pd
import pdb
from primer_summarizer_module import PRIMERSummarizer
# from Llama_summarizer_module import LlamaSummarizer
from datasets import load_dataset, load_metric
import json
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    LEDTokenizer,
    LEDForConditionalGeneration,
)
from dataloader import (
    get_dataloader_summ,
    get_dataloader_summiter,
)
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
# from pytorch_lightning.plugins import DDPPlugin
from pathlib import Path
import torch.distributed as dist
from itertools import chain


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')



def train(args):
    args.compute_rouge = True
    if args.resume_ckpt is not None:
        model = LightingSummarizer.load_from_checkpoint(args.resume_ckpt, args=args)
        print("load from checkpoint")
    else:
        model = LightingSummarizer(args)

    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = args.model_path + "summ_checkpoints/"

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-{vloss:.2f}-{avgr:.4f}",
        save_top_k=args.saveTopK,
        monitor="avgr",
        mode="max",
        save_last=False,
        save_on_train_epoch_end=False,
    )
    

    tqdm_progbar_callback = TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate * args.acc_batch)

    # initialize logger
    logger = TensorBoardLogger(args.model_path + "tb_logs", name="my_model")

    # initialize trainer
    pl.seed_everything(args.rand_seed, workers=True) # sets seeds for numpy, torch and python.random.
    trainer = pl.Trainer(
        devices=args.devices if args.devices!=0 else 'auto',
        # track_grad_norm=-1,
        max_steps=args.total_steps,
        # max_epochs=-1, # -1: unlimited training
        # use_distributed_sampler=False,
        accumulate_grad_batches=args.acc_batch,
        check_val_every_n_epoch=1 if args.num_train_data > 100 else 1, # TODO: validation each epoch, before else 5
        val_check_interval=1.0, # validate 1/0.25=4 times each epoch. # 0.25
        logger=logger,
        log_every_n_steps=1, # TODO: validation each epoch, before 5
        callbacks=[checkpoint_callback, tqdm_progbar_callback],
        enable_checkpointing=True,
        # progress_bar_refresh_rate=args.progress_bar_refresh_rate * args.acc_batch,
        enable_progress_bar=True,
        precision='16-mixed', # 32
        accelerator=args.accelerator,
        strategy = args.strategy, 
        # deterministic=True, # ensure full reproducibility
    )
    # TODO: rewrite trainer 


    # load datasets
    if args.join_method in ["multi_doc_rag"]:
        if args.dataset_name in ["multi_news"]:
            hf_datasets = load_dataset(f"HF-SSSVVVTTT/{args.dataset_name}_predict2")
        else:
            hf_datasets = load_dataset(f"HF-SSSVVVTTT/{args.dataset_name}_predict")
        train_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "train", args.num_workers, True
        )
        valid_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "validation", args.num_workers, False
        )
        test_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "test", args.num_workers, False
        )
    elif args.join_method in ["original_rank", "truncate_last_rank", "truncate_last"]:
        hf_datasets = load_dataset(f"HF-SSSVVVTTT/{args.dataset_name}_edus")
        train_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "train", args.num_workers, True
        )
        valid_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "validation", args.num_workers, False
        )
        test_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "test", args.num_workers, False
        )
    else:
        if args.dataset_name in ["multi_news", "multi_x_science_sum"]:
            if args.join_method in ["no_rand_sentence", "indoc_rand_sentence", "global_rand_sentence", 
                                    "sim_sent_transformer", "indoc_sim_sent_transformer", 
                                    "only_drop_lowsim_sent"]:
                with open('../dataset/my_processed_dataset/multi_news_sentence_dataset.json', 'r') as json_file:
                    sentence_datasets = json.load(json_file)
                if args.sent_sim_type == "sent_transformer":
                    with open('../dataset/my_processed_dataset/multi_news_sentence_similarity_SentTransformer.json', 'r') as json_file:
                        sentence_scores = json.load(json_file)
                train_dataloader = get_dataloader_summ(
                    args, sentence_datasets, model.tokenizer, "train", args.num_workers, True, 
                    sentence_scores = sentence_scores
                )
                valid_dataloader = get_dataloader_summ(
                    args, sentence_datasets, model.tokenizer, "validation", args.num_workers, False, 
                    sentence_scores = sentence_scores
                )
                test_dataloader = get_dataloader_summ(
                    args, sentence_datasets, model.tokenizer, "test", args.num_workers, False, 
                    sentence_scores = sentence_scores
                )
            else:
                hf_datasets = load_dataset(args.dataset_name, cache_dir=args.data_path)
                train_dataloader = get_dataloader_summ(
                    args, hf_datasets, model.tokenizer, "train", args.num_workers, True
                )
                valid_dataloader = get_dataloader_summ(
                    args, hf_datasets, model.tokenizer, "validation", args.num_workers, False
                )
                test_dataloader = get_dataloader_summ(
                    args, hf_datasets, model.tokenizer, "test", args.num_workers, False
                )
        elif (
            ("duc" in args.dataset_name)
            or ("tac" in args.dataset_name)
            or args.dataset_name == "wcep"
            or args.dataset_name == "wikisum"
        ):
            # 20 data from duc2003
            dataset = torch.load(args.data_path + "train.pt")
            train_dataloader = get_dataloader_summ(
                args, dataset, model.tokenizer, "train", 0, True
            )
            # 10 data from duc2003
            if os.path.exists(args.data_path + "val.pt"):
                dataset = torch.load(args.data_path + "val.pt")
            else:
                dataset = torch.load(args.data_path + "valid.pt")
            valid_dataloader = get_dataloader_summ(
                args, dataset, model.tokenizer, "validation", 0, True
            )
            dataset = torch.load(args.data_path + "test.pt")
            test_dataloader = get_dataloader_summ(
                args, dataset, model.tokenizer, "test", 0, False
            )
        elif args.dataset_name == "arxiv":
            with open(args.data_path + "train.txt", "r") as of:
                all_lines = of.readlines()
            dataset = [json.loads(l) for l in all_lines]
            train_dataloader = get_dataloader_summ(
                args, dataset, model.tokenizer, "train", 0, False
            )
            with open(args.data_path + "val.txt", "r") as of:
                all_lines = of.readlines()
            dataset = [json.loads(l) for l in all_lines]
            valid_dataloader = get_dataloader_summ(
                args, dataset, model.tokenizer, "validation", 0, False
            )
            with open(args.data_path + "test.txt", "r") as of:
                all_lines = of.readlines()
            dataset = [json.loads(l) for l in all_lines]
            test_dataloader = get_dataloader_summ(
                args, dataset, model.tokenizer, "test", 0, False
            )
    # pdb.set_trace()
    # TODO: use test for valid
    # trainer.fit(model, train_dataloader, valid_dataloader)
    trainer.fit(model, train_dataloader, test_dataloader)
    # trainer.fit(model, train_dataloader, test_dataloader, 
    #             ckpt_path="run_saves/tsy_join_method_train_arxiv_default/summ_checkpoints/step=50760-vloss=1.78-avgr=0.3073.ckpt")
    if args.test_imediate:
        args.resume_ckpt = checkpoint_callback.best_model_path
        print(args.resume_ckpt)
        if args.test_batch_size != -1:
            args.batch_size = args.test_batch_size
        args.mode = "test"
        test(args)


def test(args):
    args.compute_rouge = True
    tqdm_progbar_callback = TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)
    # initialize trainer
    trainer = pl.Trainer(
        devices=args.devices if args.devices!=0 else 'auto',
        # track_grad_norm=-1,
        # max_steps=args.total_steps * args.acc_batch,
        max_steps = -1, 
        # use_distributed_sampler=False,
        log_every_n_steps=5,
        enable_checkpointing=False, # True
        # progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        callbacks=[tqdm_progbar_callback],
        enable_progress_bar=True,
        precision='16-mixed', # 32
        accelerator=args.accelerator,
        strategy=args.strategy, 
        # limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1.0,
    )

    if args.resume_ckpt is not None:
        model = LightingSummarizer.load_from_checkpoint(args.resume_ckpt, args=args)
        print("load from checkpoint")
    else:
        model = LightingSummarizer(args)

    # load dataset
    if args.join_method in ["multi_doc_rag"]:
        if args.dataset_name in ["multi_news"]:
            hf_datasets = load_dataset(f"HF-SSSVVVTTT/{args.dataset_name}_predict2")
        else:
            hf_datasets = load_dataset(f"HF-SSSVVVTTT/{args.dataset_name}_predict")
        test_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "test", args.num_workers, False
        )
    elif args.join_method in ["original_rank", "truncate_last_rank", "truncate_last"]:
        hf_datasets = load_dataset(f"HF-SSSVVVTTT/{args.dataset_name}_edus")
        test_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "test", args.num_workers, False
        )
    else:
        if args.dataset_name in ["multi_news", "multi_x_science_sum"]:
            if args.join_method in ["no_rand_sentence", "indoc_rand_sentence", "global_rand_sentence", 
                                    "sim_sent_transformer", "indoc_sim_sent_transformer", 
                                    "only_drop_lowsim_sent"]:
                with open('../dataset/my_processed_dataset/multi_news_sentence_dataset.json', 'r') as json_file:
                    sentence_datasets = json.load(json_file)
                if args.sent_sim_type == "sent_transformer":
                    with open('../dataset/my_processed_dataset/multi_news_sentence_similarity_SentTransformer.json', 'r') as json_file:
                        sentence_scores = json.load(json_file)
                test_dataloader = get_dataloader_summ(
                    args, sentence_datasets, model.tokenizer, "test", args.num_workers, False, 
                    sentence_scores = sentence_scores
                )
            else:
                hf_datasets = load_dataset(args.dataset_name, cache_dir=args.data_path)
                test_dataloader = get_dataloader_summ(
                    args, hf_datasets, model.tokenizer, "test", 0, False
                )
        elif (
            ("duc" in args.dataset_name)
            or ("tac" in args.dataset_name)
            or args.dataset_name == "wcep"
            or args.dataset_name == "wikisum"
        ):
            if os.path.isdir(args.data_path):
                dataset = torch.load(args.data_path + "test.pt")
            else:
                dataset = torch.load(args.data_path)
            test_dataloader = get_dataloader_summ(
                args, dataset, model.tokenizer, "test", 0, False
            )
        elif args.dataset_name == "arxiv":
            with open(args.data_path + "test.txt", "r") as of:
                all_lines = of.readlines()
            dataset = [json.loads(l) for l in all_lines]
            test_dataloader = get_dataloader_summ(
                args, dataset, model.tokenizer, "test", 0, False
            )

    # test
    trainer.test(model, test_dataloader)

    # trainer.save_checkpoint("original_model/original_PRIMERA_multi_news.ckpt") # to save the original PRIMER model    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ########################
    # Gneral
    parser.add_argument("--devices", default=0, type=int, help="number of gpus to use")
    parser.add_argument(
        "--accelerator", default='gpu', type=str, help="Type of accelerator"
    ) # gpu
    parser.add_argument(
        "--strategy", default='auto', type=str, help="Whether to use ddp, ddp_spawn strategy"
    ) # gpu
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument(
        "--model_name", default="primer",
    )
    parser.add_argument(
        "--primer_path", type=str, default="allenai/PRIMERA", # ../PRIMERA/ # allenai/PRIMERA # allenai/PRIMERA-multinews
    )
    parser.add_argument("--join_method", type=str, default="tsy_design") # concat_start_wdoc_global, tsy_design
    parser.add_argument(
        "--debug_mode", action="store_true", help="set true if to debug"
    )
    parser.add_argument(
        "--compute_rouge",
        action="store_true",
        help="whether to compute rouge in validation steps",
    )
    parser.add_argument(
        "--saveRouge",
        action="store_true",
        help="whether to compute rouge in validation steps",
    )

    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)
    parser.add_argument("--model_path", type=str, default="./run_saves/tsy_join_method/") # "./pegasus/"
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--saveTopK", default=1, type=int)
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        help="Path of a checkpoint to resume from",
        default=None,
    )

    parser.add_argument("--data_path", type=str, default="../dataset/")
    parser.add_argument("--dataset_name", type=str, default="wcep") # arxiv, multi_news, multi_x_science_sum, wcep, wikisum
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for dataloader",
    )

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_tgt", default=1024, type=int)
    parser.add_argument("--min_length_tgt", default=0, type=int)
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument(
        "--adafactor", action="store_true", help="Use adafactor optimizer"
    )
    parser.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=0,
        help="seed for random sampling, useful for few shot learning",
    )

    ########################
    # TSY added
    parser.add_argument("--permute_docs", type=str_to_bool, default=False)
    parser.add_argument("--sent_sim_type", type=str, default="sent_transformer") # 
    parser.add_argument("--filter_score", type=float, default=0.0)

    ########################
    # For training
    parser.add_argument(
        "--pretrained_model_path", type=str, default="./pretrained_models/",
    )
    parser.add_argument(
        "--limit_valid_batches", type=int, default=None,
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Maximum learning rate") # 3e-5
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--accum_data_per_step", type=int, default=16, help="Number of data per step" # 16
    )
    parser.add_argument(
        "--total_steps", type=int, default=1000000, help="Number of steps to train" # 50000
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=-1,
        help="Number of training data, -1 for full dataset and any positive number indicates how many data to use",
    )

    parser.add_argument(
        "--fix_lr", action="store_true", help="use fix learning rate",
    )
    parser.add_argument(
        "--test_imediate", action="store_true", help="test on the best checkpoint",
    )
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="whether this is a run for few shot learning",
    )
    ########################
    # For testing
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="Number of batches to test in the test mode.",
    )
    parser.add_argument("--beam_size", type=int, default=1, help="size of beam search")
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1,
        help="length penalty of generated text",
    )
    parser.add_argument(
        "--mask_num",
        type=int,
        default=0,
        help="Number of masks in the input of summarization data",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=-1,
        help="batch size for test, used in few shot evaluation.",
    )
    parser.add_argument(
        "--applyTriblck",
        action="store_true",
        help="whether apply trigram block in the evaluation phase",
    )

    args = parser.parse_args()  # Get pad token id
    ####################
    if args.accum_data_per_step == -1:
        args.acc_batch = 1
    else:
        args.acc_batch = args.accum_data_per_step // args.batch_size
    args.data_path = os.path.join(args.data_path, args.dataset_name) + os.sep
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # if args.strategy == "ddp":
    #     args.devices = 4

    args.total_steps = args.total_steps//args.batch_size
    
    print(f"Current sent_sim_type: {args.sent_sim_type}")

    # if args.primer_path in ["allenai/PRIMERA-multinews"]:
    #     args.resume_ckpt = None

    if args.model_name == "primer":
        LightingSummarizer = PRIMERSummarizer
        args.primer_path = "allenai/PRIMERA"
        if args.dataset_name == "multi_news":
            args.primer_path = "allenai/PRIMERA-multinews"
            args.warmup_steps = 2500
            # args.total_steps = int(25000 * (16//args.batch_size))
            args.lr = 3e-5
        elif args.dataset_name == "multi_x_science_sum":
            args.primer_path = "allenai/PRIMERA-multixscience"
            args.warmup_steps = 2000
            # args.total_steps = int(20000 * (16//args.batch_size))
            # args.lr = 3e-5
        elif args.dataset_name == "wcep":
            args.primer_path = "allenai/PRIMERA-wcep"
            args.warmup_steps = 500
            # args.total_steps = int(5000 * (16//args.batch_size))
            # args.lr = 3e-5
        elif args.dataset_name == "arxiv":
            args.primer_path = "allenai/PRIMERA-arxiv"
            args.warmup_steps = 4000
            # args.total_steps = int(40000 * (16//args.batch_size))
            # args.lr = 3e-5
        elif args.dataset_name == "wikisum":
            args.primer_path = "./run_saves/tsy_join_method_train_wikisum_default/summ_checkpoints/step=19112-vloss=1.98-avgr=0.3459.ckpt"
    elif args.model_name == "llama":
        LightingSummarizer = LlamaSummarizer
        args.primer_path = "unsloth/llama-2-7b"


    print(args)
    with open(
        os.path.join(
            args.model_path, "args_%s_%s.json" % (args.mode, args.dataset_name)
        ),
        "w",
    ) as f:
        json.dump(args.__dict__, f, indent=2)

    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    if args.mode == "train":
        train(args)
    else:
        test(args)
