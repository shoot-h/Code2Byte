import javalang
import re
from collections import Counter
from torchtext.vocab import vocab
from pathlib import Path


def readf(a = 0,b = 32294):
    str1 = []
    str2 = []
    for i in range(a,b):#32294
        try:
            f1 = open('CoDas/CoDas_{:0>5}.java'.format(i), 'r')
            f2 = open('CoDas/CoDas_{:0>5}.class.txt'.format(i), 'r')

            data1 = f1.read()
            data2 = f2.read()
            str1.append(data1)
            str2.append(data2)
            f1.close()
            f2.close()
        except:
                pass
    
    print(len(str1))

    return str1, str2

def readwithpath(a = 0,b = 32294, path = Path('CoDas')):
    for i in range(a,b):
        try:
            f1 = open(path.joinpath('CoDas_{:0>5}.java').format(i), 'r')
            f2 = open(path.joinpath('CoDas_{:0>5}.class.txt').format(i), 'r')

            data1 = f1.read()
            data2 = f2.read()
            str1.append(data1)
            str2.append(data2)
            f1.close()
            f2.close()
        except:
                pass

    return str1, str2


def readfs(a,b):
    for i in range(a,b):
        try:
            f1 = open(path.joinpath('CoDas_{:0>5}.java').format(1), 'r')
            f2 = open(path.joinpath('CoDas_{:0>5}.class.txt').format(1), 'r')

            data1 = f1.read()
            data2 = f2.read()
            str1.append(data1)
            str2.append(data2)
            f1.close()
            f2.close()
        except:
                pass

    return str1, str2

def texttokenize(str):
    return re.findall(".*?[\t\n]", str)

def srctokenize(str):
    tokensrc = list(javalang.tokenizer.tokenize(str))
    tokens = []
    for i in range (len(tokensrc)):
        tokens.append(tokensrc[i].value)
    return tokens

def build_srcvocab(texts):
    
    counter = Counter()
    for text in texts:
        counter.update(srctokenize(text))
    return vocab(counter, specials=['<unk>', '<pad>', '<start>', '<end>'])

def build_textvocab(texts):
    
    counter = Counter()
    for text in texts:
        counter.update(texttokenize(text))
    return vocab(counter, specials=['<unk>', '<pad>', '<start>', '<end>'])



#print(texttokenize(str2[0]))

#print(srctokenize(str1[0]))