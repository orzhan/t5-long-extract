import nltk
import os, sys
from tqdm import tqdm
from difflib import SequenceMatcher
import pandas as pd

nltk.download('punkt')


ids = [x.replace('.txt','') for x in os.listdir( 'training/annual_reports' )]
gold_ids = {}
for name in os.listdir( 'training/gold_summaries'):
  id, index = name.replace('.txt','').split('_')
  if not (id in gold_ids):
    gold_ids[id] = []
  gold_ids[id].append(name)

summaries=[]
refs=[]
data=[]
data_ids=[]

content_cut_start = 22000
content_cut_end = 44000


brs=[]
ntp=[]
for id in tqdm(ids[:]):
  with open('training/annual_reports/%s.txt' % id) as fin:
    content = fin.read()
    if len(content) < 200:
      continue
    truths=[]
    truth_parts=[]
    true_blocks=[]
    true_as=[]
    true_sizes=[]
    # Find continuous gold summaries
    for name in gold_ids[id]:
      with open('training/gold_summaries/%s' % name) as tin:
        truth = tin.read()
        true_blocks.append([])
        if len(truth) < 200:
          continue
        truths.append(truth.split())
        blocks = SequenceMatcher(None, content[:content_cut_end].split(), truth.split()).get_matching_blocks()
        found = False
        for block in blocks:
          if block.size > 0:
            true_blocks[-1].append(block)
            true_as.append(block.a)
            true_sizes.append(block.size)
          if block.size >= len(truth.split())*0.66:
            truth_part = (' '.join(content[:content_cut_start].split()[block.a:block.a+block.size])).replace('\x00','').strip()
            if len(truth_part) > 100:
              shift = 0 if block.size < 2000-16 else block.size-2000+16
              content_end = (' '.join(content[:content_cut_start].split()[block.a+shift:block.a+shift+2000])).replace('\x00','').strip()
              truth_end = (' '.join(content[:content_cut_start].split()[block.a+block.size:block.a+block.size+16])).replace('\x00','').strip()
            
            if len(truth_part) > 0 and len(content) > 0 and len(content_end) > 0 and len(truth_end) > 0:
              truth_parts.append([truth_part, truth, block.a])
              found = True
              break
    # Select one gold summary with best intersection score
    true_ratings=[]
    best_rating=0
    best_truth=''
    ntp.append(len(truth_parts))
    if len(truth_parts) > 0:
      for tp in truth_parts:
        rating=0
        tps = tp[1].split()
        for pp in truths:
          blocks = SequenceMatcher(None, tps, pp).get_matching_blocks()
          for b in blocks:
            rating += b.size
        rating = rating/len(tp[1]) 
        true_ratings.append(rating)
        if rating > best_rating:
          best_rating = rating
          best_truth = tp[0]
      brs.append(best_rating)
      if len(best_truth) > 100:
        data.append(['sum: ' + content[:content_cut_start].replace('\x00','').strip(), best_truth])
        data_ids.append(id)
        
data = pd.DataFrame(data)
data.columns=['text','summary']
train_size=len(data)-200

data[:train_size].to_csv('tr.csv',index=False)
data[train_size:].to_csv('te.csv',index=False)
