#coding=utf-8
import numpy as np
import re


def clean_str(string):
    #
    #对句子相似度任务进行字符清洗
    #
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class dataset(object):
    def __init__(self,s1,s2,label):
        self._index_in_epoch = 0
        self._s1 = s1
        self._s2 = s2
        self._label = label
        self._example_nums = 6888
        self._epochs_completed = 0

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._example_nums:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._example_nums)
            np.random.shuffle(perm)
            self._s1 = self._s1[perm]
            self._s2 = self._s2[perm]
            self._label = self._label[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._example_nums
        end = self._index_in_epoch
        return np.array(self._s1[start:end]),np.array(self._s2[start:end]), np.array(self._label[start:end])

def padding_sentence(s1, s2):
    #
    # 得到句子s1,s2以后，很直观地想法就是先找出数据集中的最大句子长度，
    # 然后用<unk>对句子进行填充
    #
    s1_length_max = max([len(s.split(" ")) for s in s1])
    s2_length_max = max([len(s.split(" ")) for s in s1])
    sentence_length = max(s1_length_max, s2_length_max)
    print "已经求得最大长度sentence_length"
    for i,s in enumerate(s1):
        sen_len_cur = len(s.split(" "))
        if (sen_len_cur < sentence_length):
            reduces = sentence_length - sen_len_cur
            for reduce in range(reduces):
                s += " <unk>"
        s1[i] = s

    for i,s in enumerate(s2):
        sen_len_cur = len(s.split(" "))
        if (sen_len_cur < sentence_length):
            reduces = sentence_length - sen_len_cur
            for reduce in range(reduces):
                s += " <unk>"
        s2[i] = s
    print "9840个句子填充完毕"
    return s1, s2

def read_data_sets(train_dir):
    #
    # s1代表数据集的句子1
    # s2代表数据集的句子2
    # score代表相似度
    # s_count代表数据总共有多少行
    #
    s1 = []
    s2 = []
    score = []
    s_count = 0
    SICK_DIR = "SICK_data/SICK.txt"
    for line in open(SICK_DIR, 'r'):
        arr = line.split("\t")
        s1.append(clean_str(arr[1]))
        s2.append(clean_str(arr[2]))
        score.append(arr[4])
        s_count = s_count + 1
    # 填充句子
    s1, s2 = padding_sentence(s1, s2)
    # 引入embedding矩阵和字典
    embedding_w, embedding_dic = build_glove_dic()
    # 将每个句子中的每个单词，转化为embedding矩阵的索引
    # 如：s1_words表示s1中的单词，s1_vec表示s1的一个句子，s1_image代表所有s1中的句子
    s1_image = []
    s2_image = []
    label = []
    for i in range(s_count):
        s1_words = s1[i].split(" ")
        s2_words = s2[i].split(" ")
        s1_vec = []
        s2_vec = []
		# 如果在embedding_dic中存在该词，那么就将该词的索引加入到s1的向量表示s1_vec中，不存在则用<unk>代替
        for j,_ in enumerate(s1_words):
            if embedding_dic.has_key(s1_words[j]):
                s1_vec.append(embedding_dic[s1_words[j]])
            else:
                s1_vec.append(embedding_dic['<unk>'])
            if embedding_dic.has_key(s2_words[j]):
                s2_vec.append(embedding_dic[s2_words[j]])
            else:
                s2_vec.append(embedding_dic['<unk>'])

        s1_image.append(np.array(s1_vec))
        s2_image.append(np.array(s2_vec))
        label.append(np.array(np.array([score[i]]).astype(float)))
    
	# 求得训练集与测试集的tensor并返回
    s1_image = np.array(s1_image)
    s2_image = np.array(s2_image)
    label = np.array(label,dtype='float32')
    train_end = int(s_count*0.7)
    return s1_image[0:train_end], s2_image[0:train_end], label[0:train_end],\
           s1_image[train_end:s_count], s2_image[train_end:s_count], label[train_end:s_count],\
           embedding_w

def build_glove_dic():
    #
    # 从文件中读取pre-trained的glove文件，对应每个词的词向量
    #
    embedding_dic = {}
    embedding_w = []
    glove_dir = "glove.6B.50d.txt"
    for i, line in enumerate(open(glove_dir, 'r')):
        arr = line.split(" ")
        embedding_dic[arr[0]] = i
        embedding_w.append(np.array(arr[1:]).astype(float))
    embedding_w = np.array(embedding_w, dtype='float32')
    return embedding_w,embedding_dic

# 如果运行该文件，执行此命令，否则略过
if __name__ == "__main__":
    read_data_sets(0)