import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from pathlib import Path
import torch
from random import shuffle
import random
import os
from nltk.tokenize import sent_tokenize
import re
import sys
import spacy
from sentence_transformers import SentenceTransformer, util



def get_docs_euds_lists(docs):
    """
    get doc_clean_list and doc_edus_list from docs string.
    Args:
        docs (string): use ' || ' to split edus, and use ' ||||| ' to split documents.
    Return:
        doc_clean_list, doc_edus_list
    """
    doc_list = docs.split(' ||||| ')
    doc_clean_list = []
    doc_edus_list = []
    for i, doc in enumerate(doc_list):
        doc = doc.lstrip()
        edus = doc.split(' || ')
        doc_edus_list.append(edus)
        clean_doc = " ".join(edus)
        clean_doc = re.sub(r'\s+', ' ', clean_doc) # 替换连续多个空格为一个
        doc_clean_list.append(clean_doc)
    return doc_clean_list, doc_edus_list



class SummarizationDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        dataset_name,
        join_method,
        tokenizer,
        max_input_len,
        max_output_len,
        mask_num=5,
        num_data=-1,
        rand_seed=1,
        is_test=False,
        dataset_type="train",
        permute_docs=False,
        d_sent_score=None,
        filter_score=0.0,
    ):
        self.hf_dataset = hf_dataset
        self.dataset_name = dataset_name
        self.join_method = join_method
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        if join_method in ["concat_start_wdoc_global", "tsy_design", 
                           "no_rand_sentence", "indoc_rand_sentence", "global_rand_sentence",
                           "sim_sent_transformer", "indoc_sim_sent_transformer", "only_drop_lowsim_sent", 
                           "multi_doc_rag", "original_rank", "truncate_last_rank", "truncate_last"]:
            # self.docsep_token_id = self.tokenizer.additional_special_tokens_ids[0]
            self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        self.mask_id = self.tokenizer.mask_token_id
        self.mask_num = mask_num
        if num_data != -1 and not is_test and num_data < len(list(hf_dataset)):
            random.seed(rand_seed)
            self.hf_dataset = random.sample(list(hf_dataset), num_data)
        self.dataset_type = dataset_type
        self.permute_docs = permute_docs
        # # 加载英文模型，用于句子分割
        # self.sentence_split_model = spacy.load("en_core_web_lg")
        # self.sentence_split_model.disable_pipe("parser")
        # self.sentence_split_model.enable_pipe("senter")
        # # 加载sentence transformer，用于计算相似度
        # self.sent_trainsformer = SentenceTransformer('all-mpnet-base-v2')
        self.d_sent_score = d_sent_score
        self.filter_score = filter_score # will filter out sentence which similarity less than filter_score

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        # filtering and ranking
        if self.join_method in ["multi_doc_rag"]: # only filtering on the last doc within limited token len
            all_docs = entry["document"]
            tgt = entry["summary"]
            doc_rank = entry["predict_doc_rank"]
            doc_edu_scores = entry["predict_doc_edu_score"]

            # get multi_docs list
            doc_clean_list, doc_edus_list = get_docs_euds_lists(all_docs)
            available_token_length = self.max_input_len - (len(doc_clean_list)-1) - 2
            assert len(doc_clean_list)==len(doc_rank), "number of docs is not same as number of ranks"

            # ranking
            doc_clean_list = [doc_clean_list[i] for i in doc_rank]
            doc_edus_list = [doc_edus_list[i] for i in doc_rank]
            doc_edu_scores = [doc_edu_scores[i] for i in doc_rank]

            # encoding
            mask_num = self.mask_num
            input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
            input_ids_list = []
            all_docs_token_num = 0
            doc_num_token_list = []
            for doc_index, doc in enumerate(doc_clean_list):
                assert len(doc_edu_scores[doc_index])==len(doc_edus_list[doc_index]), "number of edu_scores not same as number of edus"
                
                # input_ids_each_doc = self.tokenizer.encode(doc)[1:-1]
                input_ids_each_doc = self.tokenizer.encode(doc)[1:-1]
                n_doc_tokens = len(input_ids_each_doc)

                doc_num_token_list.append(n_doc_tokens)
                all_docs_token_num += n_doc_tokens
                
                input_ids_list.append(input_ids_each_doc)
            
            # filtering inversely, input_ids_list[:doc_index_to_be_filtered] and the filtered doc
            if all_docs_token_num > available_token_length: # available_token_length: 4096
                # find the doc need to be filtered
                doc_index_to_be_filtered = np.searchsorted(np.cumsum(doc_num_token_list), available_token_length)
                doc_edus = doc_edus_list[doc_index_to_be_filtered]
                edu_scores = doc_edu_scores[doc_index_to_be_filtered]

                n_doc_tokens = doc_num_token_list[doc_index]
                
                doc_edus_ranks = np.argsort(edu_scores) # increase ranking
                edu_ids_to_be_filtered = []
                for edu_rank_i in doc_edus_ranks:
                    edu = doc_edus[edu_rank_i]
                    edu_ids = self.tokenizer.encode(edu.strip(), add_special_tokens=False)
                    edu_num_token = len(edu_ids)
                    all_docs_token_num = all_docs_token_num-edu_num_token
                    
                    edu_ids_to_be_filtered.append(edu_rank_i) # record which edu should be filtered
                    if all_docs_token_num <= available_token_length:
                        break
                
                doc_edus_filtered = []
                for edu_i in range(len(doc_edus)):
                    if edu_i not in edu_ids_to_be_filtered:
                        doc_edus_filtered.append(doc_edus[edu_i])
                clean_doc_filtered = " ".join(doc_edus_filtered)
                clean_doc_filtered = re.sub(r'\s+', ' ', clean_doc_filtered) # 替换连续多个空格为一个
                input_ids_each_doc = self.tokenizer.encode(clean_doc_filtered)[1:-1]
                input_ids_list = input_ids_list[:doc_index_to_be_filtered] + [input_ids_each_doc]
            

            # fitering end, assemble input_ids
            input_ids = []
            for i, final_input_ids in enumerate(input_ids_list):
                input_ids.extend(final_input_ids)
                if i != len(input_ids_list) - 1:
                    input_ids.append(self.docsep_token_id)
            
            input_ids = input_ids[:(self.max_input_len-2)]
            input_ids = ([self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id])
            assert len(input_ids) <= 4096, 'input_ids larger than 4096.'

            output_ids = self.tokenizer.encode(
                tgt, truncation=True, max_length=self.max_output_len
            )
        elif self.join_method in ["original_rank"]:
            all_docs = entry["document"]
            tgt = entry["summary"]
            doc_scores = entry["sent_qa_trans_doc_score_chunk"]
            doc_edu_scores = entry["sent_qa_trans_doc_edu_score"]
            doc_token_nums = entry["doc_token_num"]
            doc_clean_list, doc_edus_list = get_docs_euds_lists(all_docs)
            doc_rank = np.argsort(doc_scores)[::-1]

            ## ranking
            doc_clean_list = [doc_clean_list[i] for i in doc_rank]
            doc_edus_list = [doc_edus_list[i] for i in doc_rank]
            doc_edu_scores = [doc_edu_scores[i] for i in doc_rank]

            mask_num = self.mask_num
            input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
            for i, doc in enumerate(doc_clean_list):
                input_ids.extend(
                    self.tokenizer.encode(
                        doc,
                        truncation=True,
                        max_length=(self.max_input_len - mask_num) // len(doc_clean_list),
                    )[1:-1]
                )
                # input_ids.append(self.docsep_token_id)
                if i != len(doc_clean_list) - 1:
                    input_ids.append(self.docsep_token_id)
            input_ids = (
                [self.tokenizer.bos_token_id]
                + input_ids
                + [self.tokenizer.eos_token_id]
            )

            assert len(input_ids) <= 4096, 'input_ids larger than 4096.'

            output_ids = self.tokenizer.encode(
                tgt, truncation=True, max_length=self.max_output_len
            )
        elif self.join_method in ["truncate_last_rank", "truncate_last"]:
            all_docs = entry["document"]
            tgt = entry["summary"]
            doc_scores = entry["sent_qa_trans_doc_score_chunk"]
            doc_edu_scores = entry["sent_qa_trans_doc_edu_score"]
            doc_token_nums = entry["doc_token_num"]
            doc_clean_list, doc_edus_list = get_docs_euds_lists(all_docs)
            doc_rank = np.argsort(doc_scores)[::-1]
            if self.join_method in ["truncate_last"]: # rand permutation
                doc_rank = doc_rank[np.random.permutation(len(doc_rank))]

            ## ranking
            doc_clean_list = [doc_clean_list[i] for i in doc_rank]
            doc_edus_list = [doc_edus_list[i] for i in doc_rank]
            doc_edu_scores = [doc_edu_scores[i] for i in doc_rank]

            mask_num = self.mask_num
            input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
            for i, doc in enumerate(doc_clean_list):
                input_ids.extend(
                    self.tokenizer.encode(
                        doc,
                        truncation=False,
                    )[1:-1]
                )
                # input_ids.append(self.docsep_token_id)
                if i != len(doc_clean_list) - 1:
                    input_ids.append(self.docsep_token_id)
            input_ids = input_ids[:(self.max_input_len-2)]
            input_ids = (
                [self.tokenizer.bos_token_id]
                + input_ids
                + [self.tokenizer.eos_token_id]
            )

            assert len(input_ids) <= 4096, 'input_ids larger than 4096.'

            output_ids = self.tokenizer.encode(
                tgt, truncation=True, max_length=self.max_output_len
            )
        else:
            # single doc setting
            if self.dataset_name == "pubmed":
                src = entry["article"]
                tgt = entry["abstract"]
                input_ids = self.tokenizer.encode(
                    src, truncation=True, max_length=self.max_input_len
                )
                output_ids = self.tokenizer.encode(
                    tgt, truncation=True, max_length=self.max_output_len
                )
            else:  # multi-doc setting
                if self.dataset_name == "multi_news":
                    if self.join_method in ["no_rand_sentence", "indoc_rand_sentence", "global_rand_sentence", 
                                            "sim_sent_transformer", "indoc_sim_sent_transformer", 
                                            "only_drop_lowsim_sent"]:
                        all_docs = entry["document"]
                    else:
                        all_docs = entry["document"].split("|||||")# [:-1]
                        for i, doc in enumerate(all_docs):
                            doc = doc.replace("\n", " ") # TODO: 查看不将\n变为" "，将其变为分割的一部分，作为子doc或者paragraph，再根据子doc进行筛选。
                            doc = " ".join(doc.split())
                            all_docs[i] = doc
                    tgt = entry["summary"]
                elif self.dataset_name == "multi_x_science_sum":
                    all_docs = [entry["abstract"]]
                    for d in entry["ref_abstract"]["abstract"]:
                        if len(d) > 0:
                            all_docs.append(d)
                    tgt = entry["related_work"]
                    # remove all @cite_d
                    tgt = re.sub(r"\@cite_\d+", "cite", tgt)

                elif ("duc" in self.dataset_name) or ("tac" in self.dataset_name):
                    all_docs = entry["document"]
                    tgt = entry["summary"][0]  # simply use the first gt summary
                elif self.dataset_name == "wcep" or self.dataset_name == "arxiv":
                    all_docs = entry["document"]
                    tgt = entry["summary"]
                elif self.dataset_name == "wikisum":
                    all_docs = entry["text"]
                    tgt = entry["tgt"]


                if self.join_method == "plain_concat":
                    src = "\n".join(all_docs)
                    input_ids = self.tokenizer.encode(
                        src, truncation=True, max_length=self.max_input_len
                    )
                elif self.join_method == "concat_start_eachdoc":
                    input_text = []
                    for doc in all_docs:
                        length = 0
                        all_sents = sent_tokenize(doc)
                        for s in all_sents:
                            input_text.append(s)
                            length += len(s.split())
                            if length >= self.max_input_len // len(all_docs):
                                break
                    input_ids = self.tokenizer.encode(
                        " ".join(input_text),
                        truncation=True,
                        max_length=self.max_input_len,
                    )
                elif self.join_method == "concat_start_eachdoc_wsent_global":
                    input_ids = []
                    for doc in all_docs:
                        sents = [
                            " [sent] ".join(sent_tokenize(p)) + " [sent]"
                            for p in doc.split("\n")
                            if p != ""
                        ]
                        doc = "\n".join(sents)
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=self.max_input_len // len(all_docs),
                            )[1:-1]
                        )
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                elif self.join_method == "concat_start_wdoc_global":
                    mask_num = self.mask_num

                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    for i, doc in enumerate(all_docs):
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=(self.max_input_len - mask_num) // len(all_docs),
                            )[1:-1]
                        )
                        # input_ids.append(self.docsep_token_id)
                        if i != len(all_docs) - 1:
                            input_ids.append(self.docsep_token_id)
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                # no permute sentences
                elif self.join_method == "no_rand_sentence":
                    mask_num = self.mask_num
                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    
                    all_sentences = []
                    for i, doc_id in enumerate(all_docs.keys()):
                        doc_sentences = all_docs[doc_id]
                        for sentence in doc_sentences:
                            input_ids.extend(
                                self.tokenizer.encode(
                                    sentence,
                                    truncation=True,
                                    max_length=(self.max_input_len - mask_num),
                                )[1:-1]
                            )
                        if i != len(all_docs) - 1:
                            input_ids.append(self.docsep_token_id)
                    input_ids = input_ids[:(self.max_input_len-2)]
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                # in document permute sentences
                elif self.join_method == "indoc_rand_sentence":
                    mask_num = self.mask_num
                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    
                    all_sentences = []
                    for i, doc_id in enumerate(all_docs.keys()):
                        doc_sentences = all_docs[doc_id]
                        doc_sentences = random.sample(doc_sentences, len(doc_sentences))
                        for sentence in doc_sentences:
                            input_ids.extend(
                                self.tokenizer.encode(
                                    sentence,
                                    truncation=True,
                                    max_length=(self.max_input_len - mask_num),
                                )[1:-1]
                            )
                        if i != len(all_docs) - 1:
                            input_ids.append(self.docsep_token_id)
                    input_ids = input_ids[:(self.max_input_len-2)]
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                # random permute the sentences, without docsep_token_id: cross document sentences permutation
                elif self.join_method == "global_rand_sentence":
                    mask_num = self.mask_num
                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    
                    all_sentences = []
                    for doc_id in all_docs.keys():
                        doc_sentences = all_docs[doc_id]
                        all_sentences.extend(doc_sentences)
                    all_sentences = random.sample(all_sentences, len(all_sentences))
                    for sentence in all_sentences:
                        input_ids.extend(
                            self.tokenizer.encode(
                                sentence,
                                truncation=True,
                                max_length=(self.max_input_len - mask_num),
                            )[1:-1]
                        )
                    input_ids.append(self.docsep_token_id)
                    input_ids = input_ids[:(self.max_input_len-2)]
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                # drop low similarity sentence, no rerank
                elif self.join_method == "only_drop_lowsim_sent":
                    all_doc_similarities = self.d_sent_score[idx]
                    mask_num = self.mask_num
                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    
                    all_sentences = []
                    for i, doc_id in enumerate(all_docs.keys()):
                        doc_sentences = all_docs[doc_id]
                        doc_similarities = all_doc_similarities[doc_id]
                        for sent_index in range(len(doc_sentences)):
                            if doc_similarities[sent_index] > self.filter_score:
                                sentence = doc_sentences[sent_index]
                                input_ids.extend(
                                    self.tokenizer.encode(
                                        sentence,
                                        truncation=True,
                                        max_length=(self.max_input_len - mask_num),
                                    )[1:-1]
                                )
                        if i != len(all_docs) - 1:
                            input_ids.append(self.docsep_token_id)
                    
                    input_ids = input_ids[:(self.max_input_len-2)] # truncation
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                # most similar sentences on forward
                elif self.join_method == "sim_sent_transformer":
                    all_doc_similarities = self.d_sent_score[idx]
                    mask_num = self.mask_num
                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    
                    all_sentences = []
                    all_similarities = []
                    for doc_id in all_docs.keys():
                        doc_sentences = all_docs[doc_id]
                        all_sentences.extend(doc_sentences)
                        doc_similarities = all_doc_similarities[doc_id]
                        all_similarities.extend(doc_similarities)
                    
                    # rearange sentences based on the similarity
                    sorted_index = np.argsort(all_similarities)[::-1]

                    for sent_index in sorted_index:
                        if all_similarities[sent_index] > self.filter_score:
                            sentence = all_sentences[sent_index]
                            input_ids.extend(
                                self.tokenizer.encode(
                                    sentence,
                                    truncation=True,
                                    max_length=(self.max_input_len - mask_num),
                                )[1:-1]
                            )
                    input_ids.append(self.docsep_token_id)
                    input_ids = input_ids[:(self.max_input_len-2)]
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                elif self.join_method == "indoc_sim_sent_transformer":
                    all_doc_similarities = self.d_sent_score[idx]
                    mask_num = self.mask_num
                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    
                    all_sentences = []
                    for i, doc_id in enumerate(all_docs.keys()):
                        doc_sentences = all_docs[doc_id]
                        doc_similarities = all_doc_similarities[doc_id]
                        doc_sorted_index = np.argsort(doc_similarities)[::-1]
                        for sent_index in doc_sorted_index:
                            if doc_similarities[sent_index] > self.filter_score:
                                sentence = doc_sentences[sent_index]
                                input_ids.extend(
                                    self.tokenizer.encode(
                                        sentence,
                                        truncation=True,
                                        max_length=(self.max_input_len - mask_num),
                                    )[1:-1]
                                )
                        if i != len(all_docs) - 1:
                            input_ids.append(self.docsep_token_id)
                    
                    input_ids = input_ids[:(self.max_input_len-2)] # truncation
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                # indoc similarity and indoc truncation
                # filter out sentence similarity less than 0.3
                # then truncate each documents to fit the limitation of 4096
                # TODO: this part not finished yet
                elif self.join_method == "indoc_sim_sent_transformer_truncation":
                    all_doc_similarities = self.d_sent_score[idx]
                    mask_num = self.mask_num
                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    
                    all_sentences = []
                    for i, doc_id in enumerate(all_docs.keys()):
                        all_sentences.extend(all_docs[doc_id])
                    
                    suppose_inputs_id_length = len(self.tokenizer.encode(" ".join(all_sentences)))
                    suppose_truncate_length = suppose_inputs_id_length - self.max_input_len
                    
                    all_similarities = []
                    for i, doc_id in enumerate(all_docs.keys()):
                        doc_sentences = all_docs[doc_id]
                        doc_similarities = all_doc_similarities[doc_id]
                        doc_sorted_index = np.argsort(doc_similarities)[::-1]
                        for sent_index in doc_sorted_index:
                            if doc_similarities[sent_index] > self.filter_score:
                                sentence = doc_sentences[sent_index]
                                input_ids.extend(
                                    self.tokenizer.encode(
                                        sentence,
                                        truncation=True,
                                        max_length=(self.max_input_len - mask_num),
                                    )[1:-1]
                                )
                        if i != len(all_docs) - 1:
                            input_ids.append(self.docsep_token_id)
                    
                    input_ids = input_ids[:(self.max_input_len-2)] # truncation
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                # chunks ranks according to similarity
                # TODO: not finish yet
                # for example, chunk size=200, first guess the chunk num of each doc
                # then guess the containing sentence num in each chunk
                # assemble multiple sentences to a chunk
                elif self.join_method == "sim_chunks_transformer":
                    all_doc_similarities = self.d_sent_score[idx]
                    mask_num = self.mask_num
                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    
                    all_sentences = []
                    for i, doc_id in enumerate(all_docs.keys()):
                        all_sentences.extend(all_docs[doc_id])
                    
                    suppose_inputs_id_length = len(self.tokenizer.encode(" ".join(all_sentences)))
                    suppose_truncate_length = suppose_inputs_id_length - self.max_input_len
                    
                    all_similarities = []
                    for i, doc_id in enumerate(all_docs.keys()):
                        doc_sentences = all_docs[doc_id]
                        doc_similarities = all_doc_similarities[doc_id]
                        doc_sorted_index = np.argsort(doc_similarities)[::-1]
                        for sent_index in doc_sorted_index:
                            if doc_similarities[sent_index] > self.filter_score:
                                sentence = doc_sentences[sent_index]
                                input_ids.extend(
                                    self.tokenizer.encode(
                                        sentence,
                                        truncation=True,
                                        max_length=(self.max_input_len - mask_num),
                                    )[1:-1]
                                )
                        if i != len(all_docs) - 1:
                            input_ids.append(self.docsep_token_id)
                    
                    input_ids = input_ids[:(self.max_input_len-2)] # truncation
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                elif self.join_method == "tsy_design":
                    available_token_length = self.max_input_len - (len(all_docs)-1) - 2
                    input_ids_list = []
                    for doc in all_docs:
                        # input_ids_each_doc = self.tokenizer.encode(doc)[1:-1]
                        input_ids_each_doc = self.tokenizer(doc)['input_ids'][1:-1]
                        num_input_ids_doc = len(input_ids_each_doc)
                        input_ids_list.append(input_ids_each_doc)
                        # print(len(input_ids_each_doc))

                    input_ids_list = sorted(input_ids_list, key=len)

                    avg_len_each_doc = self.max_input_len//len(all_docs)
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
                    if self.permute_docs and self.dataset_type=='train':
                        final_input_ids_list = random.sample(final_input_ids_list, len(final_input_ids_list))

                    input_ids = []
                    for i, final_input_ids in enumerate(final_input_ids_list):
                        input_ids.extend(final_input_ids)
                        if i != len(final_input_ids_list) - 1:
                            input_ids.append(self.docsep_token_id)

                    input_ids = ([self.tokenizer.bos_token_id]
                                + input_ids
                                + [self.tokenizer.eos_token_id])
                    assert len(input_ids) <= 4096, 'input_ids larger than 4096.'


                output_ids = self.tokenizer.encode(
                    tgt, truncation=True, max_length=self.max_output_len
                )

        if self.tokenizer.bos_token_id is None:  # pegasus
            output_ids = [self.tokenizer.pad_token_id] + output_ids
        if self.dataset_type == "train":
            return torch.tensor(input_ids), torch.tensor(output_ids)
        else:
            return torch.tensor(input_ids), torch.tensor(output_ids), tgt


class PretrainDataset(IterableDataset):
    def __init__(
        self,
        inputs_dir,
        dataset_type,
        max_input_len,
        max_output_len,
        use_ddp=False,
        remove_masks=False,
        mask_id=0,
    ):
        super().__init__()
        if isinstance(inputs_dir, list):
            self._input_files = inputs_dir
        else:
            inputs_dir = Path(os.path.join(inputs_dir, dataset_type))
            self._input_files = [path for path in inputs_dir.glob("*.pt")]
        self.shuffle = dataset_type == "train"
        self._input_files = sorted(self._input_files)
        if self.shuffle:
            self._shuffle()
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.start = 0
        self.end = len(self._input_files)
        self.use_ddp = use_ddp
        self.remove_masks = remove_masks
        self.mask_id = mask_id

    def _loaddata(self, idx):
        file = self._input_files[idx]
        cur_data = torch.load(file)
        if self.shuffle:
            shuffle(cur_data)
        return cur_data

    def _shuffle(self):
        # shuffle the list of data files after each epoch
        shuffle(self._input_files)

    def _set_worker(self):
        # The whole dataset covering all the files in self._input_files
        overall_start = 0
        overall_end = len(self._input_files)

        # Get the worker id in the current world
        worker_info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (
            (worker_info.num_workers, worker_info.id)
            if worker_info is not None
            else (1, 0)
        )

        # Get the worker id in the overall worlds
        if self.use_ddp:
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            worker_global_rank = global_rank * num_workers + worker_id
        else:
            worker_global_rank = worker_id
            world_size = 1

        # Get the total number of workers and split tasks accordingly
        worker_world_size = num_workers * world_size
        per_worker = int(
            math.ceil((overall_end - overall_start) / float(worker_world_size))
        )

        # Set the current task, based on overall worker id and task splitting
        self.start = overall_start + worker_global_rank * per_worker
        self.end = min(self.start + per_worker, overall_end)

    def __iter__(self):
        self._set_worker()
        all_indices = list(range(self.start, self.end))
        if self.shuffle:
            # use inifinite iterators for training
            while True:
                shuffle(all_indices)
                for i in all_indices:
                    print("datafile is ", self._input_files[i])
                    sys.stdout.flush()
                    cur_data = self._loaddata(i)
                    while len(cur_data) != 0:
                        # print('data index is ', len(cur_data))
                        data = cur_data.pop()
                        if self.remove_masks:
                            data["src"] = list(
                                filter(lambda a: a != self.mask_id, data["src"])
                            )
                            # print(data["src"])
                        if len(data["src"]) > self.max_input_len:
                            data["src"] = data["src"][: (self.max_input_len - 1)] + [
                                data["src"][-1]
                            ]  # add </s>
                        if len(data["tgt"]) > self.max_output_len:
                            data["tgt"] = data["tgt"][: (self.max_output_len - 1)] + [
                                data["tgt"][-1]
                            ]  # add </s>
                        yield torch.tensor(data["src"]), torch.tensor(data["tgt"])
        else:
            # use normal iterators for validation
            for i in all_indices:
                print("datafile is ", self._input_files[i])
                sys.stdout.flush()
                cur_data = self._loaddata(i)
                while len(cur_data) != 0:
                    # print('data index is ', len(cur_data))
                    data = cur_data.pop()
                    if self.remove_masks:
                        data["src"] = list(
                            filter(lambda a: a != self.mask_id, data["src"])
                        )
                        # print(data["src"])
                    if len(data["src"]) > self.max_input_len:
                        data["src"] = data["src"][: (self.max_input_len - 1)] + [
                            data["src"][-1]
                        ]  # add </s>
                    if len(data["tgt"]) > self.max_output_len:
                        data["tgt"] = data["tgt"][: (self.max_output_len - 1)] + [
                            data["tgt"][-1]
                        ]  # add </s>
                    yield torch.tensor(data["src"]), torch.tensor(data["tgt"])


class SummarizationIterDataset(IterableDataset):
    def __init__(
        self,
        join_method,
        dataset_name,
        tokenizer,
        inputs_dir,
        dataset_type,
        max_input_len,
        max_output_len,
        use_ddp=False,
        mask_num=0,
    ):
        super().__init__()
        if isinstance(inputs_dir, list):
            self._input_files = inputs_dir
        else:
            inputs_dir = Path(os.path.join(inputs_dir, dataset_type))
            self._input_files = [path for path in inputs_dir.glob("*.pt")]
        self._input_files = sorted(self._input_files)
        self.shuffle = dataset_type == "train"
        if self.shuffle:
            self._shuffle()
        self.join_method = join_method
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.start = 0
        self.end = len(self._input_files)
        self.use_ddp = use_ddp
        if join_method == "concat_start_wdoc_global":
            self.docsep_token_id = self.tokenizer.additional_special_tokens_ids[0]
        self.mask_num = mask_num
        self.dataset_type = dataset_type

    def _loaddata(self, idx):
        file = self._input_files[idx]
        cur_data = torch.load(file)
        if self.shuffle:
            shuffle(cur_data)
        return cur_data

    def _shuffle(self):
        # shuffle the list of data files after each epoch
        shuffle(self._input_files)

    def _set_worker(self):
        # The whole dataset covering all the files in self._input_files
        overall_start = 0
        overall_end = len(self._input_files)

        # Get the worker id in the current world
        worker_info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (
            (worker_info.num_workers, worker_info.id)
            if worker_info is not None
            else (1, 0)
        )

        # Get the worker id in the overall worlds
        if self.use_ddp:
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            worker_global_rank = global_rank * num_workers + worker_id
        else:
            worker_global_rank = worker_id
            world_size = 1

        # Get the total number of workers and split tasks accordingly
        worker_world_size = num_workers * world_size
        per_worker = int(
            math.ceil((overall_end - overall_start) / float(worker_world_size))
        )

        # Set the current task, based on overall worker id and task splitting
        self.start = overall_start + worker_global_rank * per_worker
        self.end = min(self.start + per_worker, overall_end)

    def __iter__(self):
        self._set_worker()

        for i in range(self.start, self.end):
            print("datafile is ", i)
            cur_data = self._loaddata(i)
            while len(cur_data) != 0:
                # print("data index is ", len(cur_data))
                data = cur_data.pop()
                all_docs = data["text"]
                if self.join_method == "plain_concat":
                    src = "\n".join(all_docs)
                    tgt = data["tgt"]
                    input_ids = self.tokenizer.encode(
                        src, truncation=True, max_length=self.max_input_len
                    )
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )
                elif self.join_method == "concat_start_eachdoc":
                    input_ids = []
                    for doc in all_docs:
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=self.max_input_len // len(all_docs),
                            )[1:-1]
                        )
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                    tgt = data["tgt"]
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )
                elif self.join_method == "concat_start_eachdoc_wsent_global":
                    input_ids = []
                    for doc in all_docs:
                        sents = [
                            " [sent] ".join(sent_tokenize(p)) + " [sent]"
                            for p in doc.split("\n")
                            if p != ""
                        ]
                        doc = "\n".join(sents)
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=self.max_input_len // len(all_docs),
                            )[1:-1]
                        )
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                    tgt = data["tgt"]
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )
                elif self.join_method == "concat_start_wdoc_global":
                    mask_num = self.mask_num
                    tgt = data["tgt"]
                    # src='<mask>'*10+src
                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    for doc in all_docs:
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=(self.max_input_len - mask_num)
                                // len(all_docs),
                            )[1:-1]
                        )
                        input_ids.append(self.docsep_token_id)
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )

                if self.tokenizer.bos_token_id is None:  # pegasus
                    output_ids = [self.tokenizer.pad_token_id] + output_ids
                    input_ids = input_ids[1:]
                if self.dataset_type == "train":
                    yield torch.tensor(input_ids), torch.tensor(output_ids)
                else:
                    yield torch.tensor(input_ids), torch.tensor(output_ids), tgt


def collate_fn(batch, model_name='primer'):
    # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
    # primera: pad: 1, eos: 2, sos: 0;
    # pegasus: pad: 0, eos: 1
    # llama2: pad: 0, eos: 2, sos: 1
    if model_name == "primer":
        if batch[0][0][-1].item() == 2: # eos_token_id
            pad_token_id = (
                1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
            )
        elif batch[0][0][-1].item() == 1:
            pad_token_id = (
                0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
            )
        else:
            assert False
    elif model_name == "llama":
        pad_token_id = (0)
    else:
        assert False
    
    train = True
    if len(batch[0]) == 3:
        train = False
        tgt = [item[2] for item in batch]
        batch = [item[:2] for item in batch]
    input_ids, output_ids = list(zip(*batch))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    if input_ids.shape[1]>4096:
        print(input_ids.shape)
    padding_len = (512 - input_ids.shape[1] % 512) % 512
    input_ids = torch.nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)

    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    if train:
        return input_ids, output_ids
    else:
        return input_ids, output_ids, tgt


def get_dataloader_summ(
    args, hf_datasets, tokenizer, split_name, num_workers, is_train, sentence_scores=None
):
    if args.join_method in ["multi_doc_rag", "original_rank", "truncate_last_rank", "truncate_last"]:
        d = hf_datasets[split_name]
    else:
        if (
            ("duc" in args.dataset_name)
            or ("tac" in args.dataset_name)
            or args.dataset_name == "wcep"
            or args.dataset_name == "wikisum"
        ):
            d = hf_datasets
        elif args.dataset_name == "arxiv":

            d = [
                {
                    "document": [" ".join(s) for s in single_data["sections"]],
                    "summary": " ".join(
                        [
                            sent.replace("<S>", "").replace("</S>", "").strip()
                            for sent in single_data["abstract_text"]
                        ]
                    ),
                }
                for single_data in hf_datasets
            ]
        else:
            d = hf_datasets[split_name]
    if sentence_scores != None:
        d_sent_score = sentence_scores[split_name]
    else:
        d_sent_score = None
    # d = d.select(range(24)) # TODO: d变为20的样本，只用于测试
    dataset = SummarizationDataset(
        hf_dataset=d,
        dataset_name=args.dataset_name,
        join_method=args.join_method,
        tokenizer=tokenizer,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        mask_num=args.mask_num,
        num_data=args.num_train_data,
        rand_seed=args.rand_seed,
        is_test=(split_name == "test"),
        dataset_type=split_name,
        permute_docs=args.permute_docs, 
        d_sent_score=d_sent_score,
        filter_score=args.filter_score,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        # sampler=sampler,
        collate_fn=lambda x: collate_fn(x, args.model_name),
    )


def get_dataloader_pretrain(
    args, inputs_dir, dataset_type, num_workers, use_ddp=False, mask_id=0
):
    dataset = PretrainDataset(
        inputs_dir,
        dataset_type,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        use_ddp=use_ddp,
        remove_masks=args.remove_masks,
        mask_id=mask_id,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def get_dataloader_summiter(
    args, tokenizer, inputs_dir, dataset_type, num_workers, use_ddp=False
):
    dataset = SummarizationIterDataset(
        args.join_method,
        args.dataset_name,
        tokenizer,
        inputs_dir,
        dataset_type,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        use_ddp=use_ddp,
        mask_num=args.mask_num,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
