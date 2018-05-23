

# tmp_1 = """E. Relative mRNA expression in proximal hind limb muscle of AR21Q 
# (n = 5) and AR113Q (n = 3) males backcrossed to C57BL/6J. ** p<0.01, ***p<0.001 
# by Student's t test. F. Relative mRNA expression in spinal cord of AR21Q (n = 5) 
# and AR113Q (n = 3) males (mean +/− SEM). n. s. = not significant by Student's t test."""

# tmp_2 = """(b) Titration of phospholipids (for example, TOCL) to LC3 was 
# performed to evaluate the ratio that prevents 50% of the LC3 from 
# entering the gels (IC50) as an index of relative affinity. 
# Bottom left inset: representative gel for TOCL/LC3 = 12; 
# top right inset: comparison of IC50 values for TOCL versus 
# dioleoyl-phosphatidic acid (DOPA) and tetralinoleoyl-CL (TLCL) 
# versus monolyso-trilinoleoyl-CL (lyso-CL). The IC50 for DOPG was >15. 
# *P0.05 versus TOCL; †P0.05 versus TLCL."""

# tmp = "A, B Quantitative analysis of (A) TUNEL-positive cells and (B) caspase-3-positive cells in IHC in the spleens of CLP + PBS, CLP + Pal-Scram #1, and CLP + Smaducin-6mice. Three independent experiments (n = 3 mice per group per experiments) were performed. At least five hot spots in a section of TUNEL and IHC per experiment were selected, and average count was determined. The data were expressed as a mean percentage of total cell numbers and statistically analyzed by a t-test and show the mean ± SD of three independent experiments. **P < 0.005, ***P < 0.001 compared to sham or vehicle control (CLP + Pal-Scram #1)."

# for specific_symbol in "!\"#$%'()*+,-./:;<=>?@[\\]_`{|}~":     # °C ^
#     tmp2 = tmp.replace(specific_symbol, ' '+specific_symbol+' ')
# print('tmp2: ' + tmp2)

# tmp3 = ' '.join(tmp.split())

# for specific_symbol in "!\"#$%'()*+,-./:;<=>?@[\\]_`{|}~":     # °C ^
#     tmp3 = tmp3.replace(specific_symbol, ' '+specific_symbol+' ')
# print(tmp3)

# tmp2 = tmp2.replace('   ', ' ').replace('  ', ' ')
# tmp3 = tmp3.replace('   ', ' ').replace('  ', ' ')
# print(tmp2)
# print(tmp3)

# if '' in tmp2.split():
#     print('df')
# if '  ' in tmp3:
#     print('dfdf')

# from keras.layers import Dense, Flatten, Input
# from keras.layers.embeddings import Embedding
# from keras.models import Model
# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import one_hot
# # define documents
# docs = ['Well done!',
#         'Good work',
#         'Great effort',
#         'nice work',
#         'Excellent!',
#         'Weak',
#         'Poor effort!',
#         'not good',
#         'poor work',
#         'Could have done better.']
# # define class labels
# labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
# # integer encode the documents
# vocab_size = 50
# encoded_docs = [one_hot(d, vocab_size) for d in docs]
# print(encoded_docs)
# # pad documents to a max length of 4 words
# max_length = 4
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)
# # define the model
# input = Input(shape=(4, ))
# x = Embedding(vocab_size, 8, input_length=max_length)(input)
# # 单独做一个embedding模型，利于后面观察
# embedding = Model(input,x)
# x = Flatten()(x)
# x = Dense(1, activation='sigmoid')(x)
# model = Model(inputs=input, outputs=x)
# # compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# # summarize the model
# print(model.summary())

# '''
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         (None, 4)                 0
# _________________________________________________________________
# embedding_1 (Embedding)      (None, 4, 8)              400
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 32)                0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 33
# =================================================================
# Total params: 433
# Trainable params: 433
# Non-trainable params: 0
# _________________________________________________________________
# '''

# # fit the model
# model.fit(padded_docs, labels, epochs=50, verbose=0)
# # evaluate the model
# loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# print('Accuracy: %f' % (accuracy * 100))

# # embedding的输出
# print(embedding.predict(padded_docs).shape) # (10, 4, 8)
# print(embedding.predict(padded_docs))


a = ''
del a
gc.collection()

'''
激活函数的选择
一般来说，我们的选择顺序可以理解为：
ELU > leaky ReLU (以及其变种) > ReLU > tanh > logistic
'''