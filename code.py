import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/contradictory-my-dear-watson'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#install following dependices

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,TFBertModel
import matplotlib.pyplot as plt
import tensorflow as tf
import gradio as ml

##################################################################################################################################################################################
#import train.csv file

train=pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")
print(train.info())
print(train.head())
train.drop("id",axis=1,inplace=True)
	
#count the values of entailment,neutral,contradiction..

count=train["label"].value_counts().sort_index()
label_names=["entailment","neutral","contradiction"]
count.index=label_names

print(count)


#visualize through barh graphs and pie charts

custom_colors=["#F8766D","#00BA38","#619CFF"]

count.plot.barh(color=custom_colors)
plt.xlabel("labels")
plt.ylabel("count")
plt.title("number of entries per label")
plt.show()

unique=train.nunique()
print("total languages")

print(unique["language"])


language_count=train["language"].value_counts(sort=True,ascending=False)

custom_color=["#F8766D", "#E58700", "#C99800", "#A3A500", "#6BB100", "#00BA38", "#00BF7D","#00C0AF", "#00BCD8","#00B0F6", "#619CFF", "#B983FF","#E76BF3", "#FD61D1", "#FF67A4"]

language_count.plot.pie(colors=custom_colors,
	autopct="%1.1f%%",
	pctdistance=1.0,
	labeldistance=1.23,
	startangle=70)
plt.title("percentage of different laguages")
plt.show()


#######################################################################################################################################################################################################
#model initlization of pretrained model

"""
model_name = 'bert-base-multilingual-uncased'
max_length =234
dropout=0.3
tokenizer = BertTokenizer.from_pretrained(model_name)
"""
# you can also try with above pretrined model

model_name = 'bert-base-multilingual-cased'
max_length =234
dropout=0.3
tokenizer = BertTokenizer.from_pretrained(model_name)

#tokenization of sentences

def tokenize_sentences(sentence):
    tokens=list(tokenizer.tokenize(sentence))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)

#encoder with input sentences of hypothesis,premises,tokenizer


def bert_encoder(hypotheses, premises, tokenizer):

        num_examples = len(hypotheses)

        sentence1 = tf.ragged.constant([
          tokenize_sentences(sentence)
          for sentence in np.array(hypotheses)])
        sentence2 = tf.ragged.constant([
          tokenize_sentences(sentence)
           for sentence in np.array(premises)])

        cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
        input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)[:,:max_length]

        input_mask = tf.ones_like(input_word_ids).to_tensor()

        type_cls = tf.zeros_like(cls)
        type_s1 = tf.zeros_like(sentence1)
        type_s2 = tf.ones_like(sentence2)
        input_type_ids = tf.concat(
          [type_cls, type_s1, type_s2], axis=-1).to_tensor()[:,:max_length]

        inputs = {
          'input_words_ids': input_word_ids.to_tensor(),
          'input_mask': input_mask,
          'input_type_ids': input_type_ids}

        return inputs

#let us train the bert_encoder model

trainer=bert_encoder(train.premise.values,train.hypothesis.values,tokenizer)
print(trainer)

#length of input token ids of each words in a sentences

length=trainer["input_words_ids"].shape[-1]
print(f"length of input_words_ids:{length}")

#here we are used Tensorflowbert model rather than hugging face ..or etc..
#construction of input deep neurons 

bert_encoder=TFBertModel.from_pretrained(model_name)
input_words_ids=tf.keras.Input(shape=(length),dtype=tf.int32,name="input_words_ids")
input_mask=tf.keras.Input(shape=(length),dtype=tf.int32,name="input_mask")
input_type_ids=tf.keras.Input(shape=(length),dtype=tf.int32,name="input_type_ids")

#taking embeddings 
embeddings=bert_encoder([input_words_ids,input_mask,input_type_ids])[0]

#output layer of three 
output=tf.keras.layers.Dense(3,activation="softmax")(embeddings[:,0,:])

#compile the model with different learning rates 

model=tf.keras.Model(inputs=[input_words_ids,input_mask,input_type_ids],outputs=output)
model.compile(tf.keras.optimizers.Adam(learning_rate=1e-6),loss="sparse_categorical_crossentropy",metrics=["accuracy"])

#training starts now .we trained this model via distributed training with gpu's and gpu p 100's
#were you also view how to apply distributed training and parallel processing in official websited of Tensorflow.org



strategy=tf.distribute.get_strategy()
print(f"number of replicas:{strategy.num_replicas_in_sync}")

with strategy.scope():
    model.summary()
  
"""
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_words_ids (InputLayer)   [(None, 234)]        0           []                               
                                                                                                  
 input_mask (InputLayer)        [(None, 234)]        0           []                               
                                                                                                  
 input_type_ids (InputLayer)    [(None, 234)]        0           []                               
                                                                                                  
 tf_bert_model (TFBertModel)    TFBaseModelOutputWi  177853440   ['input_words_ids[0][0]',        
                                thPoolingAndCrossAt               'input_mask[0][0]',             
                                tentions(last_hidde               'input_type_ids[0][0]']         
                                n_state=(None, 234,                                               
                                 768),                                                            
                                 pooler_output=(Non                                               
                                e, 768),                                                          
                                 past_key_values=No                                               
                                ne, hidden_states=N                                               
                                one, attentions=Non                                               
                                e, cross_attentions                                               
                                =None)                                                            
                                                                                                  
 tf.__operators__.getitem (Slic  (None, 768)         0           ['tf_bert_model[0][0]']          
 ingOpLambda)                                                                                     
                                                                                                  
 dense (Dense)                  (None, 3)            2307        ['tf.__operators__.getitem[0][0]'
                                                                 ]                                
                                                                                                  
==================================================================================================
Total params: 177,855,747
Trainable params: 177,855,747
Non-trainable params: 0
__________________________________________________________________________________________________
"""

#check whether the gpu is initialized or not

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2304)])  # Set the memory limit according to your GPU's capacity
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

#training the model
history=model.fit(trainer,train.label.values,epochs=30,verbose=1,batch_size=20,validation_split=0.2)
pd.DataFrame(history.history).plot(figsize=(12,6))

