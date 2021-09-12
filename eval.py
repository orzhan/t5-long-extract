import os
from tqdm import tqdm
import pandas as pd

summaries=[]
refs=[]
ids = [x.replace('.txt','') for x in os.listdir( 'testing/annual_reports' )]
data=[]
data_ids=[]
data_long=[]

# Convert data to csv
for id in tqdm(ids[:]):
  with open('testing/annual_reports/%s.txt' % id) as fin:
    content = fin.read().replace('\x00','').strip()
    data.append([content[:22000], 'X'])
    data_long.append(content)
    data_ids.append(id)
    
data = pd.DataFrame(data)
data.columns=['text','summary']
data.to_csv('tet.csv',index=False)

os.system('''mkdir verification''')
os.system('''mkdir verification/system''')

# Run the model

os.system('''python transformers/examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path orzhan/t5-long-extract \
    --source_prefix "sum: " \
    --do_predict \
    --validation_file tet.csv \
    --test_file tet.csv \
    --output_dir tst-summarization \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --max_source_length 4096 \
    --max_target_length 64''')
    
ids = [x.replace('.txt','') for x in os.listdir( 'testing/annual_reports' )]

# Create summaries based on predicted sequences
with open('tst-summarization/generated_predictions.txt','r') as fin:
  preds=fin.readlines()
summaries=[]
aa=[]
trues=[]

for i, row in tqdm(data.iterrows(), total=len(data)):
  pos=None
  # Split text into words
  tt=data_long[i]
  tt_words=['']
  tt_spaces=[]
  prev='a'
  for k in range(0,len(tt)):
    if tt[k].isspace() and prev.isspace():
      tt_spaces[-1] += tt[k]
    elif tt[k].isspace() and not prev.isspace():
      tt_spaces.append(tt[k])
    elif not tt[k].isspace() and prev.isspace():
      tt_words.append(tt[k])
    else:
      tt_words[-1] += tt[k]
    prev = tt[k]
  tt_spaces.append('')
  # Locate prediction in the text (best match)
  pp=preds[i].split()
  posmax=len(pp)
  for p in range(min(len(tt_words)-len(pp), 4000)):
    fail=0
    for j in range(len(pp)):
      if tt_words[p+j] != pp[j]:
        fail+=1         # count different words
    if fail<posmax:     # found new best match
      pos=p
      posmax=fail
      if fail==0:
        break
  if pos is None:
    pos=0
  aa.append(pos)
  predicted=''
  pred_length=1000
  for j in range(pos, pos+pred_length):             # take 1000 words from pos
    predicted += tt_words[j] + tt_spaces[j]
    if j == len(tt_words)-1:
      break
  summaries.append(predicted)
  with open('verification/system/%s_%s.txt' % (data_ids[i],'summary'),'w') as fout:
    fout.write(predicted)