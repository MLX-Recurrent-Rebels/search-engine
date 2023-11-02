#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset


# In[2]:


from datasets import load_dataset
import pandas as pd
dataset = load_dataset("ms_marco", "v1.1")
train_data = dataset["train"]
train_df = pd.DataFrame(train_data)


# In[4]:


print(train_data)


# In[5]:


#print(train_df)
print(train_df.iloc[2])


# In[ ]:






# In[6]:


queries = train_df['query'].tolist()

passage_container = train_df['passages'].tolist()
passage_texts = []



for item in passage_container:
    for passage_text in item["passage_text"]:
        passage_texts.append(passage_text)

unique_passage_list = list(set(passage_texts))


with open('corpus.txt', 'w') as file:
    for query in queries:
        file.write(f'{query}\n')
    for unique_text in unique_passage_list:
        file.write(f'{unique_text}\n')


# In[14]:


import sentencepiece as spm

model_prefix = 'tokenizer'  # Prefix for the model files
vocab_size = 10000  # Number of tokens in the vocabulary
model_type = 'unigram'  # You can choose 'unigram' or 'bpe'

# Train the SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type
)


# In[15]:


# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load(f'{model_prefix}.model')

# Tokenize a sentence
#query_tokens = []
#passage_tokens = []
with open('query_tokens_uni.txt', 'w') as query_tokens:
     for query in queries:
        query_tokens.write(f'{sp.encode_as_pieces(query)}\n')
      #   query_tokens.append(sp.encode_as_pieces(query))
with open('passage_tokens_uni.txt', 'w') as passage_tokens:
     for unique_text in unique_passage_list:
        passage_tokens.write(f'{sp.encode_as_pieces(unique_text)}\n')



# In[16]:





# In[17]:


print(unique_passage_list[0])


# In[ ]:




