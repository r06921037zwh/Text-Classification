#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:40:25 2018

@author: zhewei
"""
import os
import pandas as pd
import numpy as np
import pickle
import jieba
from collections import Counter
keywords = ['<PAD>', '<UNK>', '<NUM>']

def merge_file_on_chatId():
    print("Merging File on ChatID ...")
    df = pd.read_excel('detail.xlsx')[['ChatId', 'Time', 'Who', 'Content']]
    # drop Chatbot content
    #df = df[df['Who'] != 'Chatbot']
    #df.dropna(thresh=1, inplace=True)
    #df.sort_values(['ChatId', 'Time'], inplace=True)
    # reset index
    #df.reset_index(drop=True, inplace=True)
    #head = wochatbot.head()

    #readin mapping
    mapping = pd.read_excel('EchatWorkCode.xlsx', sheet_name='符合chatId')
    df_ = pd.merge(df, mapping, how='inner', on='ChatId')
    df_.sort_values(['ChatId', 'Time'], inplace=True)
    df_.reset_index(drop=True, inplace=True)
    return df_

def get_the_paragraph(df):
    print("Get the Paragraph ...")
    #get the paragraph
    idx = 0
    start = 0
    end = 0
    records = []
    while(True):
        start = idx
        end = idx + 1
        if(end >= len(df)):
            break
        
        while( (df.loc[start]['ChatId'] == df.loc[end]['ChatId']) ):
            end = end + 1
            if(end >= len(df)): break
        records.append([start, end - 1, df.loc[start]['ChatId']])
        idx = end
        #print(end)
    return records

def make_text_labels_pair(df, records):
    print("Make text label pair ...")
    # extract text and labels
    x, y = [], []
    texts = []
    labels = []
    for record in records:
        Set = set()
        text = ""
        for txt in df.loc[record[0]: record[1]]['Content']:
            Set.add(txt)
        for txt in list(Set):
            text = text + str(txt) + " "    
        texts.append(text)
        
        label = set()
        for workcode in df.loc[record[0]: record[1]]['WorkCode']:
            label.add(workcode)
        labels.append(list(label))
        
        # make data pair
        for i in list(label):
            x.append(text)
            y.append(i)

        #pickle.dump(x, open('x.pickle', 'wb'))    
        #pickle.dump(y, open('y.pickle', 'wb'))
    return x, y

def load_pickle():
    print("Load file ...")
    X = pickle.load(open('x.pickle', 'rb'))    
    y = pickle.load(open('y.pickle', 'rb'))
    return X, y
    
    
    
def seg_str(x, stopwords):
    print("Segamenting String ...")
    X = []
    for sentence in x:
        seg_list = jieba.cut(sentence)
        processed = []
        for seg in seg_list:
            if seg not in stopwords and seg != " ":
                processed.append(seg)
        X.append(processed)
    return X

def encoding(X, y, word2id, code2label):
    print("Encoding Raw Data ...")
    x_outputs = []
    for sentence in X:
        x_output = []
        for word in sentence:
            if word in word2id:
                x_output.append(word2id[word])
            else:
                x_output.append(word2id['<UNK>'])
        x_outputs.append(x_output)
    
    y_outputs = []
    for code in y:
        y_outputs.append(code2label[code])
        
    return x_outputs, y_outputs
    
def make_dict(X, y, num_of_words=5000):
    print("Making Dictionary ...")
    # make word2id and id2word dictionary
    words = []
    for sentence in X:
        words.extend(sentence)
    
    # replace all digit with "NUM"
    words_ = []
    for word in words:
        if word.replace('.','',1).isdigit() == True:
            words_.append("<NUM>")
        else:
            words_.append(word)
            
    counter = Counter(words_).most_common(num_of_words)
    vocab_list = keywords + [w for w, _ in counter]
    word2id = {k: v for v, k in list(enumerate(vocab_list))}
    id2word = {k: v for k, v in list(enumerate(vocab_list))}

    
    # make code2label and label2code dictionary
    workcode = pd.read_excel('EchatWorkCode.xlsx', sheet_name='Class')
    code2label = dict(zip(list(workcode['Code']), list(workcode['label'])))
    label2code = dict(zip(list(workcode['label']), list(workcode['Code'])))
    
    return word2id, id2word, code2label, label2code
    
    '''
    dictionary = {}
    for word in vocab_list:
        dictionary[str(word)] = model[str(word)]
    
    print("Saving the dictionary ...")
    pickle.dump(dictionary, open('dictionary', 'wb'))
    return dictionary
    '''
def cal_len(X):
    len_list = []
    for x in X:
        len_list.append(len(x))
    return len_list

def zero_padding(X, length=250):
    X_ = []
    for x in X:
        x_ = []
        if len(x) > length:
            x_ = x[:length]
        else:
            x_ = x + [0] * (length - len(x))
        X_.append(x_)
    return X_

def save_all(word2id, id2word, code2label, label2code, X_padding, y_encoded, X_len):
    print("Saving All File ...")
    dic = {'word2id': word2id, 
           'id2word': id2word,
           'code2label': code2label,
           'label2code': label2code}
    
    data = {'X': X_padding,
            'y': y_encoded,
            'X_len': X_len}
    
    pickle.dump(dic, open(os.path.join('processed_data','dic.pkl'), 'wb'))
    pickle.dump(data, open(os.path.join('processed_data','data.pkl'), 'wb'))
    
def main():
    df = merge_file_on_chatId()
    records = get_the_paragraph(df)
    X, y = make_text_labels_pair(df, records)
    #X, y =  load_pickle()
    
    # segment str into words
    stopwords = open('stopwords.txt', 'r', encoding='utf-8').read().split('\n')
    X = seg_str(X, stopwords)
    
    # make dictionary and encoding 
    word2id, id2word, code2label, label2code = make_dict(X, y)
    X_encoded, y_encoded = encoding(X, y, word2id, code2label)
    
    # calculate length of each sentence
    X_len = cal_len(X_encoded)
    
    # zero padding to the same length
    X_padding = zero_padding(X_encoded)
    
    # save all into a pickle
    save_all(word2id, id2word, code2label, label2code, X_padding, y_encoded, X_len)

if __name__ == '__main__':
    main()
    
'''
import seaborn as sns
ax = sns.distplot(np.array(y_encoded))
ax.set(xlabel='Class', ylabel='Probability')
'''