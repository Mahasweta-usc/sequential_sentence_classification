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
MAX_LEN = 256

# def process_(text):
#   text = text.replace("\r\n"," ").replace("\n"," ")
#   text = " ".join(re.sub('>',"",text).split())
#   return text

# def sent_break(text):
#     doc = nlp(text)
#     lines = [line.text for line in doc.sentences]
#     return lines

# def prune():
#     global self.candidate[url]
#     temp = []
#     for elem in self.candidate[url]:
#         if any(len(elem) < len(cand) and set(elem).issubset(set(cand)) for cand in self.candidate[url]):
#             temp.append(elem)

#     for elem in temp: self.candidate[url].remove(elem)

# def single_entries():
#   global self.candidate[url]
#   for idx, elem in enumerate(self.candidate[url]):
#     if len(elem) == 1:
#       if MAX_LEN > 2*len(tokenizer.tokenize(elem[0])):
#         self.candidate[url][idx].append(elem[0])
#       else:
#         str_len = int(len(elem[0])/2)
#         self.candidate[url][idx] = [elem[0][:str_len]]
#         self.candidate[url][idx].append(elem[0][str_len:])

# def segmenter():
#   global self.candidate[url]
#   global email_sent[url]
#   sum_ = 0;
#   lim = int(0.9*MAX_LEN)
#   self.candidate[url].append([])
#   for idx, elem in enumerate(email_sent[url]):
#     remain = lim - sum_
#     sum_ += len(tokenizer.encode(elem))
#     if sum_ > lim:
#       retain = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[:remain]))
#       carryover = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[remain:]))
#       if not idx:
#         email_sent[url][idx] = carryover
#         email_sent[url].insert(idx,retain)
#         self.candidate[url][-1].append(retain) 
#         # return idx + 1
#       return idx
    
#     self.candidate[url][-1].append(elem)


# def email_to_json():
#     global json_data, self.candidate[url]
#     for idx, row in enumerate(self.candidate[url]):
#       entry = dict()
#       entry["sentences"] = row
#       entry["labels"] = ["0"]*len(row)
#       entry["abstract_id"] = 0
#       json_data.append(entry)



# json_data, self.candidate[url], email_sent[url] = [], [], []

import multiprocessing
from multiprocessing import pool
from multiprocessing.pool import ThreadPool



@Predictor.register('SeqClassificationPredictor')
class SeqClassificationPredictor(Predictor):
    """
    Predictor for the abstruct model
    """

    def process_(self,text):
      text = text.replace("\r\n"," ").replace("\n"," ")
      text = " ".join(re.sub('>',"",text).split())
      return text

    def sent_break(self,text):
      doc = nlp(text)
      sentences = [line.text for line in doc.sentences]
      lines = [line for line in doc.sentences]
      verbs = []
      prev = 0
      for idx, line in enumerate(lines):
        tags = [word.upos for word in line.words if word.upos in ['AUX','VERB']]
        if tags:
          if idx - prev <= 1: 
            print(sentences[prev:idx+1],"-----"," ".join(sentences[prev:idx+1]))
            verbs.append(" ".join(sentences[prev:idx+1]))
          else:
            for elem in sentences[prev:idx+1]: verbs.append(elem)
          prev = idx + 1

      for elem in sentences[prev:]: verbs.append(elem)
      return verbs


    def prune(self,url):
        temp = []
        for elem in self.candidate[url]:
            if any(len(elem) < len(cand) and set(elem).issubset(set(cand)) for cand in self.candidate[url]):
                temp.append(elem)

        for elem in temp: self.candidate[url].remove(elem)

    def single_entries(self,url):
        for idx, elem in enumerate(self.candidate[url]):
          if len(elem) == 1:
            if MAX_LEN > 2*len(tokenizer.tokenize(elem[0])):
              self.candidate[url][idx].append(elem[0])
            else:
              str_len = int(len(elem[0])/2)
              self.candidate[url][idx] = [elem[0][:str_len]]
              self.candidate[url][idx].append(elem[0][str_len:])

    def segmenter(self,url):
        sum_ = 0;
        lim = int(0.9*MAX_LEN)
        self.candidate[url].append([])
        for idx, elem in enumerate(email_sent[url]):
          remain = lim - sum_
          sum_ += len(tokenizer.encode(elem))
          if sum_ > lim:
            retain = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[:remain]))
            carryover = tokenizer.convert_tokens_to_string(tokenizer.tokenize(elem[remain:]))
            if not idx:
              email_sent[url][idx] = carryover
              email_sent[url].insert(idx,retain)
              self.candidate[url][-1].append(retain) 
              # return idx + 1
            return idx
          
          self.candidate[url][-1].append(elem)


    def email_to_json(self,url):
        json_data = []
        entry = dict()
        entry[url]["sentences"] = self.candidate[url]
        entry[url]["labels"] = [["0"]*len(row) for row in self.candidate[url]]
        entry[url]["abstract_id"] = 0
        json_data.append(entry)


    def segment_text(self,chunk,url):
      print(chunk)
      self.email_sent[url] = [sent_elem for sent_elem in self.sent_break(process_(chunk["last_reply"]))] 
      self.candidate[url] = []

      while True:
        pos = self.segmenter(url)
        if len(self.email_sent[url]) > 1: self.email_sent[url].pop(0)
        else: break

      self.candidate[url] = [cand for cand in self.candidate[url] if cand]
      self.prune(url)
      self.single_entries(url)
      if len(self.candidate[url]) > 500 or not len(self.candidate[url]): return []
      self.email_to_json();print(len(self.candidate[url]))
      return json_data

    def process_text(self, filename):
      n_cpu = multiprocessing.cpu_count() - 1
      f = pd.read_csv(filename,lineterminator='\n');f.drop_na(inplace=True)
      f = f[f.folder.isin["dev","user","users","announce"]]
      cols = f.columns.tolist() + ['last_reply','IS_count','IS_']
      outtable = pd.DataFrame(columns = cols)
      row_count = 0

      for i,chunk in f.iterrows():
        email = EmailReplyParser.parse_reply(chunk["message"].replace('.>','\n>'))
        chunk["last_reply"] = email
        chunk['IS_count'] = 0
        chunk['IS_'] = ""
        outtable.loc[len(outtable.index)] = chunk

      deadpool = ThreadPool(n_cpu)
      results = []
      for _ ,row in outtable.iterrows():
        results.append(self.segment_text(row,row["url"]))

      # deadpool.close()
      # deadpool.join()
      out = [r for r in results]
      out = {k:v for key,values in x.items() for x in out}
      return out, outtable
  
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        print("Enter filename: ")
        filename = input()
        outfile = filename.replace(".csv","_IS.csv")
        json_data,out = self.process_text(filename)
        self.candidate = {}
        self.email_sent = {}
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



