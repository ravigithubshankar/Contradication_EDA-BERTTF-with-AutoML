# Contradication_EDA-BERTTF-with-AutoML
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Natural language processing (NLP) has grown increasingly elaborate over the past few years. Machine learning models tackle question answering, text extraction, sentence generation, and many other complex tasks. But, can machines determine the relationships between sentences, or is that still left to humans? If NLP can be applied between sentences, this could have profound implications for fact-checking, identifying fake news, analyzing text, and much more.

If you have two sentences, there are three ways they could be related: one could entail the other, one could contradict the other, or they could be unrelated. Natural Language Inferencing (NLI) is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) are related.

Your task is to create an NLI model that assigns labels of 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) to pairs of premises and hypotheses. To make things more interesting, the train and test set include text in fifteen different languages.

                 for this we are used pretrained models of Bert-uncased and Bert-cased models BERT (Bidirectional Encoder Representations from Transformers),We have explored the difference between BERT                        cased and BERT uncased. BERT uncased and BERT cased are different in terms of BERT training using case of text in WordPiece tokenization step and presence of accent markers.

let me explain with little example about Bert-base-uncased and cased :

                                                     BERT uncased and cased in tokenization

In BERT uncased, the text has been lowercased before WordPiece tokenization step while in BERT cased, the text is same as the input text (no changes).

For example,

                    if the input is "OpenGenus", then it is converted to "opengenus" for BERT uncased while BERT cased takes in "OpenGenus".

Today, the most common approaches to NLI problems include using embeddings and transformers like BERT.in this  we trained a BERT model for how pairs of sentences (consisting of a premise and a hypothesis) are related.  via Distributed training of Gpu's and Gpu p 100's rather than cpu and tpu's

with
                                                    
                                                    Total params: 177,855,747

for testing we tested model with separate dataset.csv with different languages  you can explore it in test.csv file ..

deployment cases we used Gradio interface .taking users input via gradio interface and predicting the results .you view deployment file which is available 

________________________________________________________________________________________________________________________________________________________________________________________________________________

We'll use the following command to launch training:

                                                !tensorflow & pytorch scripts/code_train.py \
                                                       --model bert-base-uncased \
                                                       --model bert-base-mutilingual-cased\
                                                       --dataset_path train.csv \
                                                       --lr 1e-4 \
                                                       --per_device_train_batch_size 32 & 64 \
                                                       --epochs 10

the training was completed and achieved better results 

                                             

🛠 frameworks and tools used:

<img align="left" alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />


<img align="left" alt="Tensorflow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" />

<img align="left" alt="Pandas" src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" />


<img align="left" alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" />

<img align="left" alt="Scikit-learn" src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" />

<img align="left" alt="Gradio" src="https://gradio.app/" />
