#! -*- coding: utf-8 -*-
import gensim
import numpy as np
import jieba
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

# stop_text = open('stop_list.txt', 'r')
# stop_word = []
# for line in stop_text:
#     stop_word.append(line.strip())
TaggededDocument = gensim.models.doc2vec.TaggedDocument


def get_corpus():
    with open("news.txt", 'r', encoding='utf-8') as doc:
        docs = doc.readlines()
    train_docs = []
    for i, text in enumerate(docs):
        # text = ' '.join(jieba.cut(text, cut_all=True))
        word_list = text.split(' ')
        length = len(word_list)
        word_list[length - 1] = word_list[length - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        train_docs.append(document)
    return train_docs


def train(x_train, size=200, epoch_num=70):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=epoch_num)
    model_dm.save('model_doc2vec')
    return model_dm


def predict(sentence, top_k=10):
    model_dm = Doc2Vec.load("model_doc2vec")
    text_cut = jieba.cut(sentence)
    text_raw = []
    for i in list(text_cut):
        text_raw.append(i)
    inferred_vector_dm = model_dm.infer_vector(text_raw)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=top_k)
    return sims


if __name__ == '__main__':
    x_train = get_corpus()
    model_dm = train(x_train)
    sims = predict(sentence='经现场相关人员了解雨天路滑视线模糊加之车辆速度过快导致了该起事故的发生')
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print(sim, len(sentence[0]), words)
        # print(sim, len(sentence[0]))
