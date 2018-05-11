import os
import numpy as np
import pickle as pkl
from sklearn import svm
from sklearn.preprocessing import StandardScaler

'''
参考：http://www.voidcn.com/article/p-pgruxmdu-bra.html
参考：https://xacecask2.gitbooks.io/scikit-learn-user-guide-chinese-version/content/sec1.4.html
'''


# from sklearn import svm
# from sklearn import datasets
# clf = svm.LinearSVC(verbose=True)
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# print(X.shape)
# clf.fit(X, y)
# test = np.random.randint(1,10, (150, 4))
# print(clf.predict(test))


if os.path.exists('train.pkl'):
    with open('train.pkl', "rb") as f:
        x, y = pkl.load(f)
        x = np.array(x)
        y = np.array(y)
        print(x.shape)  # (114904, 212)
        print(list(y).count(1))  # 44548
        print(list(y).count(0))  # 181550
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
else:
    raise FileNotFoundError

clf = svm.SVC(kernel='linear', verbose=True, probability=True, max_iter=10000)  # polynomial  指示进度的输出
# clf = svm.LinearSVC(verbose=True)  # polynomial  指示进度的输出
clf.fit(x, y)

print('SVM模型训练完毕，保存')

# 保存Model(注:save文件夹要预先建立，否则会报错)
with open('clf.pkl', 'wb') as f:
    pkl.dump(clf, f)

#读取Model
with open('clf.pkl', 'rb') as f:
    clf2 = pkl.load(f)
    #测试读取后的Model
    test = np.random.uniform(-0.1, 0.1, 212)
    test2 = np.random.uniform(-0.1, 0.1, 212)
    cls = clf2.predict_proba([test, test2]) # [[ 0.76984457  0.23015543]]
    print(cls)