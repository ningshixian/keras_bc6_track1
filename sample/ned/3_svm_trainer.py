import os
import numpy as np
import pickle as pkl
from sklearn import svm
from sklearn import datasets


if os.path.exists('train.pkl'):
    with open('train.pkl', "rb") as f:
        x, y = pkl.load(f)
        x = np.array(x)
        y = np.array(y)
        print(x.shape)  # (114904, 212)
        print(y.shape)  # (114904,)
else:
    raise FileNotFoundError

clf = svm.SVC(kernel='linear')  # polynomial
clf.fit(x, y)


# clf = svm.SVC()
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# print(type(X))
# print(type(y))
# print(X.shape)
# print(y.shape)
# clf.fit(X,y)

# 保存Model(注:save文件夹要预先建立，否则会报错)
with open('clf.pkl', 'wb') as f:
    pkl.dump(clf, f)

# #读取Model
# with open('save/clf.pickle', 'rb') as f:
#     clf2 = pkl.load(f)
#     #测试读取后的Model
#     print(clf2.predict([[2., 2.]]))