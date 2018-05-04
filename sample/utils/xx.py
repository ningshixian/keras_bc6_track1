# a = [3,4,1,7,2]

# b=sorted(enumerate(a), key=lambda x:x[1],reverse=True)
# c1 = [x[1] for x in b]
# c2 = [x[0] for x in b]

# print(c1)
# print(c2)

dataPath = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/train'
trainPath = dataPath+ '/' + 'train.out.txt'
outPath = dataPath+ '/' + 'label1.txt'

test_path = r'/Users/ningshixian/Desktop/BC6_Track1/test_corpus_20170804/test'
BioC_PATH = r'/Users/ningshixian/Desktop/BC6_Track1/test_corpus_20170804/caption_bioc'
testPath = test_path+ '/' + 'test.out.txt'
outPath = test_path+ '/' + 'label.txt'

num=0
sentence = []
temp = []
res = []
with open(testPath, 'r') as f:
    for line in f:
        if not line=='\n':
            temp.append(line.split('\t')[-1].strip('\n'))
            sentence.append(line.split('\t')[0].strip('\n'))
        else:
            num+=1
            if num==147:
                print(sentence)
            res.append(temp)
            temp = []
            sentence = []
            
with open(outPath, 'w') as f:
    for line in res:
        f.write(''.join(line))
        f.write('\n')
