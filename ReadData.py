import numpy as np
import matplotlib.pyplot as plt


def read_data(datafile,separator=","):
    f=open(datafile)
    line = f.readline()
    labels = []
    all_data = []
    while line:
        info = line.strip("\n").split(separator)
        class_label = info[-1]
        labels.append(class_label)
        data = [float(i) for i in info]
        data[-1] = data[-1]-1
        all_data.append(data)
        line= f.readline()
    f.close()
    labels = list(set(labels))
    np.random.shuffle(all_data)
    all_data = np.asarray(all_data)
    X = all_data[:,:-1]
    Y= all_data[:,-1]
    return X,Y,labels

def plot_scatter(data,labels,figname='Class Plot'):
    colors = ['red','green','blue','cyan','magenta','yellow','orange']
    plt.style.use('dark_background')
    for d in range(len(data)):
        class_label = int(labels[d])-1
        color = colors[class_label]
        plt.scatter(data[d][0],data[d][1],c=color)
    plt.title(figname)
    plt.savefig(figname + ".png")
    plt.show()

def load_words():
    with open('Data/words_alpha.txt') as word_file:
        valid_words = list(set(word_file.read().split()))
    return valid_words

def filter_words(all_words,max_chars=8,min_chars=1):
    f = open('Data/train_words.txt','w')
    for w in all_words:
        wordlen = len(w)
        if(wordlen<=max_chars)and(wordlen>=min_chars):
            f.write("%s,%d\n"%(w,wordlen))
    f.close()

def load_characters(charfile):
    chars = {}
    idx = {}
    charlist = []
    f = open(charfile)
    line = f.readline()
    k = 0
    while line:
        info = line.strip("\n")
        chars[info] = k
        idx[k] = info
        charlist.append(info)
        k = k + 1
        line = f.readline()
    f.close()
    return chars,idx,charlist

def make_batch_of_words(words,wordlens,charmap):
    max_len = max(wordlens)
    pad_words = []
    for wi in range(len(words)):
        w_idx = [charmap[ch] for ch in words[wi]]
        w_length = wordlens[wi]
        to_pad = max_len - w_length
        pad_v = []
        for p in range(to_pad):
            pad_v.append(charmap['*'])
        if(to_pad>0):
            w_idx.extend(pad_v)
        pad_words.append(w_idx)
    pad_words = np.asarray(pad_words)
    return pad_words

def load_words_from_file(wordfile):
    words = []
    word_lengths = []
    f = open(wordfile)
    line = f.readline()
    while line:
        info = line.strip("\n").split(",")
        words.append(info[0])
        word_lengths.append(int(info[-1]))
        line = f.readline()
    f.close()
    return words,word_lengths

def plot_character_probability(prediction,characters):
    chars = np.arange(len(prediction))
    plt.bar(chars,prediction)
    plt.xticks(chars,characters)
    plt.savefig('NGram.png')
    plt.show()

# X,Y,labels = read_data('Data/spiral.txt',separator="\t")
# plot_scatter(X,Y,figname='Spiral')
# words = load_words()
# filter_words(words,min_chars=3)

# words, wl = load_words_from_file("Data/train_words.txt")
# charmap = load_characters("Data/characters.txt")
# pwords = make_batch_of_words(words[1756:1773],wl[1756:1773],charmap)
# print(pwords)
