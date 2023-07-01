#now test the model with separate csv file
test=pd.read_csv("/kaggle/input/contradictory-my-dear-watson/test.csv")
print(test.head())
#test_encoder = bert_encoder(test.premise.values, test.hypothesis.values, tokenizer)

#initilaze the pretrained model

tokenizer1 = BertTokenizer.from_pretrained(model_name)
#encoder

def bert_encoder1(hypotheses, premises, tokenizer1):

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

test_encoder=bert_encoder1(test.premise.values,test.hypothesis.values,tokenizer1)

#length of word tokens in sentences

test_encoder["input_words_ids"].shape[-1]

#predictions

predictions=[np.argmax(i)for i in model.predict(test_encoder)]

#you can preditions now
print(predictions)
