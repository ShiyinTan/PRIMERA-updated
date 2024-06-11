# %%
import argparse
import random
from transformers import (
    AutoTokenizer,
    LEDForConditionalGeneration,
)
from datasets import load_dataset
import torch

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
parser.add_argument("--original", default='False', type=str_to_bool) 
parser.add_argument("--permute_docs", default='False', type=str_to_bool) 
parser.add_argument("--beam", default=5, type=int) 

parser.add_argument(
    "--primer_path", type=str, default="allenai/PRIMERA-multinews", # ../PRIMERA/ # allenai/PRIMERA
)

parser.add_argument(
    "--strategy", default='auto', type=str, help="Whether to use ddp, ddp_spawn strategy"
) # gpu

args = parser.parse_args()
########################



# %%
dataset=load_dataset('multi_news')

# %%
PRIMER_path='allenai/PRIMERA-multinews'


########################
# Gneral
if args.original:
    primer_model_lighting = PRIMERSummarizer(args=args) # original primer
    print("original model")
else:
    checkpoint_path = './script/run_saves/tsy_join_method_train_3/summ_checkpoints/step=19677-vloss=1.88-avgr=0.3152.ckpt'
    primer_model_lighting = PRIMERSummarizer.load_from_checkpoint(checkpoint_path = checkpoint_path, args=args)
    print("checkpoint_path: ", checkpoint_path)
TOKENIZER = primer_model_lighting.tokenizer
MODEL = primer_model_lighting.model

# TOKENIZER = AutoTokenizer.from_pretrained(PRIMER_path)
# MODEL = LEDForConditionalGeneration.from_pretrained(PRIMER_path)

MODEL.cuda()
PAD_TOKEN_ID = TOKENIZER.pad_token_id
DOCSEP_TOKEN_ID = TOKENIZER.convert_tokens_to_ids("<doc-sep>")

# %%
def process_document(documents):
    input_ids_all=[]
    for data in documents:
        all_docs = data.split("|||||")
        for i, doc in enumerate(all_docs):
            doc = doc.replace("\n", " ")
            doc = " ".join(doc.split())
            all_docs[i] = doc

        #### concat with global attention on doc-sep
        input_ids = []
        for i, doc in enumerate(all_docs):
            input_ids.extend(
                TOKENIZER.encode(
                    doc,
                    truncation=True,
                    max_length=4096 // len(all_docs),
                )[1:-1]
            )
            if i != len(all_docs) - 1:
                input_ids.append(DOCSEP_TOKEN_ID)
        input_ids = (
            [TOKENIZER.bos_token_id]
            + input_ids
            + [TOKENIZER.eos_token_id]
        )
        input_ids_all.append(torch.tensor(input_ids))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_all, batch_first=True, padding_value=PAD_TOKEN_ID
    )
    return input_ids


def process_document_tsy(documents, permute_docs=False):
    input_ids_all=[]
    for data in documents:
        all_docs = data.split("|||||")
        for i, doc in enumerate(all_docs):
            doc = doc.replace("\n", " ")
            doc = " ".join(doc.split())
            all_docs[i] = doc

        #### concat with global attention on doc-sep
        
        available_token_length = 4096 - (len(all_docs)-1) - 2
        input_ids_list = []
        input_ids = []
        for i, doc in enumerate(all_docs):
            input_ids_each_doc = TOKENIZER(doc)['input_ids'][1:-1]
            num_input_ids_doc = len(input_ids_each_doc)
            input_ids_list.append(input_ids_each_doc)
        
        input_ids_list = sorted(input_ids_list, key=len)

        avg_len_each_doc = 4096//len(all_docs)
        final_input_ids_list = []
        i = 0
        while i < len(input_ids_list):
            input_ids_each_doc = input_ids_list[i]
            num_input_ids_doc = len(input_ids_each_doc)
            if num_input_ids_doc <= avg_len_each_doc:
                available_token_length -= num_input_ids_doc
                final_input_ids_list.append(input_ids_each_doc)
                # print("append: ", num_input_ids_doc, "available: ", available_token_length, "avg: ", avg_len_each_doc)
                i += 1
            else:
                avg_len_each_doc = available_token_length//(len(input_ids_list)-i)
                if num_input_ids_doc > avg_len_each_doc:
                    available_token_length -= avg_len_each_doc
                    final_input_ids_list.append(input_ids_each_doc[:avg_len_each_doc])
                    # print("append: ", num_input_ids_doc, "available: ", available_token_length, "avg: ", avg_len_each_doc)
                    i += 1
        # TODO: 使用 sample 函数生成新的随机排列列表
        if permute_docs:
            final_input_ids_list = random.sample(final_input_ids_list, len(final_input_ids_list))

        input_ids = []
        for i, final_input_ids in enumerate(final_input_ids_list):
            input_ids.extend(final_input_ids)
            if i != len(final_input_ids_list) - 1:
                input_ids.append(DOCSEP_TOKEN_ID)

        input_ids = (
            [TOKENIZER.bos_token_id]
            + input_ids
            + [TOKENIZER.eos_token_id]
        )
        assert len(input_ids) <= 4096, 'input_ids larger than 4096.'
        input_ids_all.append(torch.tensor(input_ids))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_all, batch_first=True, padding_value=PAD_TOKEN_ID
    )
    if input_ids.shape[1]>4096:
        print(input_ids.shape)
    padding_len = (512 - input_ids.shape[1] % 512) % 512
    input_ids = torch.nn.functional.pad(input_ids, (0, padding_len), value=PAD_TOKEN_ID)
    return input_ids


def batch_process(batch, tsy=False, permute_docs=False):
    if tsy:
        input_ids=process_document_tsy(batch['document'], permute_docs=permute_docs).cuda()
    else:
        input_ids=process_document(batch['document']).cuda()

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
    result={}
    result['generated_summaries'] = generated_str
    result['gt_summaries']=batch['summary']
    return result


data_idx = range(len(dataset['test']))
# data_idx = random.choices(range(len(dataset['test'])),k=1000)
dataset_small = dataset['test'].select(data_idx)


par_str = ""
if args.tsy:
    par_str = par_str + "tsy"
if args.permute_docs:
    par_str = par_str + "permute"
par_str = par_str+f"beam_{args.beam}_{args.par_id}"


result_all = dataset_small.map(lambda d: batch_process(d, tsy=args.tsy, permute_docs=args.permute_docs), batched=True, batch_size=2)
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
