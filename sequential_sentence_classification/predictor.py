from typing import List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import argparse
import jsonlines
import os, sys
import random
import json
import numpy as np
import regex as re
from fuzzywuzzy import fuzz, process
import itertools
from itertools import combinations, compress
from nltk.tokenize import sent_tokenize,word_tokenize
import pandas as pd
import stanza
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize')
from email_reply_parser import EmailReplyParser
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = int(os.environ["SENT_MAX_LEN"]); print(MAX_LEN)
from text_process import process_text

# def process_(text):
#   text = text.replace("\r\n"," ").replace("\n"," ")
#   text = " ".join(re.sub('>',"",text).split())
#   return text

# def sent_break(text):
#     doc = nlp(text)
#     lines = [line.text for line in doc.sentences]
#     return lines

# def prune():
#     global candidate
#     temp = []
#     for elem in candidate:
#         if any(len(elem) < len(cand) and set(elem).issubset(set(cand)) for cand in candidate):
#             temp.append(elem)

#     for elem in temp: candidate.remove(elem)

# def single_entries():
#   global candidate
#   for idx, elem in enumerate(candidate):
#     if len(elem) == 1:
#       if MAX_LEN > 2*len(tokenizer.tokenize(elem[0])):
#         candidate[idx].append(elem[0])
#       else:
#         str_len = int(len(elem[0])/2)
#         candidate[idx] = [elem[0][:str_len]]
#         candidate[idx].append(elem[0][str_len:])

# def segmenter():
#   global candidate
#   global email_sent
#   sum_ = 0;
#   lim = int(0.9*MAX_LEN)
#   candidate.append([])
#   for idx, elem in enumerate(email_sent):
#     remain = lim - sum_
#     sum_ += len(tokenizer.encode(elem))
#     if sum_ > lim:
#       retain = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[:remain]))
#       carryover = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[remain:]))
#       if not idx:
#         email_sent[idx] = carryover
#         email_sent.insert(idx,retain)
#         candidate[-1].append(retain) 
#         # return idx + 1
#       return idx
    
#     candidate[-1].append(elem)


# def email_to_json():
#     global json_data, candidate
#     for idx, row in enumerate(candidate):
#       entry = dict()
#       entry["sentences"] = row
#       entry["labels"] = ["0"]*len(row)
#       entry["abstract_id"] = 0
#       json_data.append(entry)



# json_data, candidate, email_sent = [], [], []

@Predictor.register('SeqClassificationPredictor')
class SeqClassificationPredictor(Predictor):
    """
    Predictor for the abstruct model
    """
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        parser = argparse.ArgumentParser(description='ArgumentParser')
        parser = self.add_subparser("filename", parser)
        args = parser.parse_args()
        filename = args.filename
        outfile = filename.replace(".csv","_IS.csv")
        json_data,out = process_text(filename)

        for idx,row in out:
            url = row["url"]
            sentences = json_data[url]["sentences"]
            labels = json_data[url]["labels"]
            predictions = []

            for sentence, label in zip(sentences,labels):
                instances = self._dataset_reader.text_to_instance(sentences=sentence,labels=label)
                output = self._model.forward_on_instances([instances])
                # print(output)
                idx = output[0]['action_probs'].argmax(axis=1).tolist()
                logits = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
                binary_labels = [int(item.split("_")[0]) for item in logits]
                predictions += list(itertools.compress(sentence,binary_labels))

            pred_out = list(set(predictions))
            row["IS_"] = "<Institutional>".join(pred_out)
            row["IS_count"] = len(pred_out)

            if predictions: print(idx,predictions)
            out.to_csv(outfile,index=False)

        exit()





# from typing import List
# from overrides import overrides

# from allennlp.common.util import JsonDict, sanitize
# from allennlp.data import Instance
# from allennlp.predictors.predictor import Predictor
# import jsonlines

# import os
# file_path = os.environ["file_path"]

# @Predictor.register('SeqClassificationPredictor')
# class SeqClassificationPredictor(Predictor):
#     """
#     Predictor for the abstruct model
#     """
#     def predict_json(self, json_dict: JsonDict) -> JsonDict:
#         pred_labels = []
#         sentences = json_dict['sentences']
#         paper_id = json_dict['abstract_id']
#         try:
#           labels = json_dict['labels']
#         except:
#           labels = [["1"]*len(sent) for sent in sentences]
#         print(sentences,labels)
#         instance = self._dataset_reader.text_to_instance(sentences=sentences,labels=labels)
#         output = self._model.forward_on_instances([instance])
#         # print(output)
#         idx = output[0]['action_probs'].argmax(axis=1).tolist()
#         # print(idx)
#         labels = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
#         # print(labels)
#         pred_labels.extend(labels)
#         assert len(pred_labels) == len(sentences)
#         preds = list(zip(sentences, pred_labels))


#         with jsonlines.open(file_path, mode='a') as writer:
#           json_dict["predictions"] = pred_labels
#           writer.write(json_dict)
#         return paper_id, preds



  