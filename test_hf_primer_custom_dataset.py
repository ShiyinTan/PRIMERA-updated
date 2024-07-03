# %%
import argparse
import json
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LEDForConditionalGeneration,
)
from script.dataloader import get_dataloader_summ
from datasets import load_dataset
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from script.primer_summarizer_module import PRIMERSummarizer



def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser()

########################
# Gneral
parser.add_argument("--par_id", default=0, type=str, help="notices for runing")
parser.add_argument(
    "--accelerator", default='gpu', type=str, help="Type of accelerator"
) # gpu
parser.add_argument("--tsy", default='False', type=str_to_bool)
parser.add_argument("--original", default='True', type=str_to_bool) 
parser.add_argument("--beam", default=5, type=int) 

parser.add_argument(
    "--primer_path", type=str, default="allenai/PRIMERA-multinews", # ../PRIMERA/ # allenai/PRIMERA
)

parser.add_argument(
    "--strategy", default='auto', type=str, help="Whether to use ddp, ddp_spawn strategy"
) # gpu
parser.add_argument("--dataset_name", type=str, default="multi_news") # arxiv
parser.add_argument("--join_method", type=str, default="no_rand_sentence") # concat_start_wdoc_global, no_rand_sentence, global_rand_sentence, indoc_rand_sentence, sim_sent_transformer 
parser.add_argument("--max_length_input", default=4096, type=int)
parser.add_argument("--max_length_tgt", default=1024, type=int)
parser.add_argument("--mask_num", type=int, default=0)
parser.add_argument("--num_train_data", type=int, default=-1)
parser.add_argument("--rand_seed", type=int, default=0)
parser.add_argument("--permute_docs", default='False', type=str_to_bool)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--sent_sim_type", type=str, default="sent_transformer")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for dataloader")

args = parser.parse_args()
########################

print(args)


# %%
PRIMER_path='allenai/PRIMERA-multinews'


########################
# Gneral
if args.original:
    # primer_model_lighting = PRIMERSummarizer(args=args) # original primer
    print("original model")
    # TOKENIZER = primer_model_lighting.tokenizer
    # MODEL = primer_model_lighting.model
    TOKENIZER = AutoTokenizer.from_pretrained(PRIMER_path)
    MODEL = LEDForConditionalGeneration.from_pretrained(PRIMER_path)
else:
    # checkpoint_path = './script/run_saves/tsy_join_method_train_global_rand_sentence/summ_checkpoints/step=11244-vloss=2.04-avgr=0.3064.ckpt'
    # checkpoint_path = './script/run_saves/tsy_join_method_train_sentence/summ_checkpoints/step=2811-vloss=1.96-avgr=0.3158.ckpt'
    # checkpoint_path = './script/run_saves/tsy_join_method_train_sim_sent_transformer/summ_checkpoints/step=11244-vloss=2.05-avgr=0.3142.ckpt'
    checkpoint_path = './script/run_saves/tsy_join_method_train_indoc_sent_transformer/summ_checkpoints/step=11244-vloss=2.05-avgr=0.3098.ckpt'

    print("checkpoint_path: ", checkpoint_path)
    primer_model_lighting = PRIMERSummarizer.load_from_checkpoint(checkpoint_path = checkpoint_path, args=args)
    TOKENIZER = primer_model_lighting.tokenizer
    MODEL = primer_model_lighting.model

# TOKENIZER = AutoTokenizer.from_pretrained(PRIMER_path)
# MODEL = LEDForConditionalGeneration.from_pretrained(PRIMER_path)

MODEL.cuda()
PAD_TOKEN_ID = TOKENIZER.pad_token_id
DOCSEP_TOKEN_ID = TOKENIZER.convert_tokens_to_ids("<doc-sep>")





# %%
# load dataset
# dataset=load_dataset('multi_news')
if args.join_method in ["concat_start_wdoc_global", "tsy_design"]:
    dataset=load_dataset('multi_news')
    # train_dataloader = get_dataloader_summ(
    #     args, dataset, TOKENIZER, "train", args.num_workers, True
    # )
    # valid_dataloader = get_dataloader_summ(
    #     args, dataset, TOKENIZER, "validation", args.num_workers, False
    # )
    test_dataloader = get_dataloader_summ(
        args, dataset, TOKENIZER, "test", args.num_workers, False
    )
else:
    with open('./dataset/my_processed_dataset/multi_news_sentence_dataset.json', 'r') as json_file:
        sentence_datasets = json.load(json_file)
    if args.sent_sim_type == "sent_transformer":
        with open('./dataset/my_processed_dataset/multi_news_sentence_similarity_SentTransformer.json', 'r') as json_file:
            sentence_scores = json.load(json_file)
    # train_dataloader = get_dataloader_summ(
    #     args, sentence_datasets, TOKENIZER, "train", args.num_workers, True, 
    #     sentence_scores = sentence_scores
    # )
    # valid_dataloader = get_dataloader_summ(
    #     args, sentence_datasets, TOKENIZER, "validation", args.num_workers, False, 
    #     sentence_scores = sentence_scores
    # )
    test_dataloader = get_dataloader_summ(
        args, sentence_datasets, TOKENIZER, "test", args.num_workers, False, 
        sentence_scores = sentence_scores
    )


result_all = {}
result_all['generated_summaries'] = []
result_all['gt_summaries'] = []
for batch in tqdm(test_dataloader):
    # print(batch)

    input_ids=batch[0].cuda()
    target = batch[2]

    # get the input ids and attention masks together
    global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device).cuda()
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    global_attention_mask[input_ids == DOCSEP_TOKEN_ID] = 1
    generated_ids = MODEL.generate(
        input_ids=input_ids,
        global_attention_mask=global_attention_mask,
        use_cache=True,
        max_length=1024,
        num_beams=args.beam,
    )
    generated_str = TOKENIZER.batch_decode(
        generated_ids.tolist(), skip_special_tokens=True
    )
    for generated_str_item, target_item in zip(generated_str, target):
        result_all['generated_summaries'].append(generated_str_item)
        result_all['gt_summaries'].append(target_item)
    # print(len(result_all['generated_summaries']))

par_str = ""
par_str = par_str+f"beam_{args.beam}_{args.par_id}"

with open(f"test_hf_save/generated_summaries_{par_str}.txt", 'w') as wf1, open(f"test_hf_save/gt_summaries_{par_str}.txt", 'w') as wf2: 
    for generated_summary in result_all['generated_summaries']:
        wf1.write(generated_summary + '\n')
    for gt_summary in result_all['gt_summaries']:
        wf2.write(gt_summary+ '\n')


from datasets import load_metric

rouge = load_metric("rouge")
with open(f"test_hf_save/generated_summaries_{par_str}.txt") as f:
    generated_summaries = []
    for line in f:
        generated_summaries.append(line.strip())
with open(f"test_hf_save/gt_summaries_{par_str}.txt") as f:
    gt_summaries = []
    for line in f:
        gt_summaries.append(line.strip())
result = rouge.compute(predictions=generated_summaries, references=gt_summaries, rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
print("ROUGE scores:")
print(result)



