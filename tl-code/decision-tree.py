"""
决策树
"""
import copy
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd


class DecisionTree(object):
    """决策树 数据结构

     Parameters:
    -----------
    col: int, default(-1)
        当前使用的第几列数据

    val: int or float or str, 分割节点
        分割节点的值,
        int or float : 使用大于进行比较
        str : 使用等于模式

    LeftChild: DecisionTree
        左子树, <= val

    RightChild: DecisionTree
        右子树, > val

    results:

    """
    def __init__(self, col: int=-1,
                 val=None,
                 left_child=None,
                 right_child=None,
                 result=None):
        self.col = col
        self.val = val
        self.left_child = left_child
        self.right_child = right_child
        self.result = result


class DecisionTreeClassifier(object):
    """使用基尼系数的分类决策树接口

    Parameters:
    -----------
    max_depth: int or None, optional(default=None)
        表示决策树的最大深度. None: 表示不设置深度,可以任意扩展,
        直到叶子节点的个数小于min_samples_split个数.

    min_samples_split : int, optional(default=2)
        表示最小分割样例数.
        if int, 表示最小分割样例树,如果小于这个数字,不在进行分割.

    min_samples_leaf : int, optional (default=1)
        表示叶节点最少有min_samples_leaf个节点树,如果小于等于这个数,直接返回.
        if int, min_samples_leaf就是最小样例数.

    min_impurity_decrease : float, optional (default=0.)
        分割之后基尼指数大于这个数,则进行分割.
        N_t / N * (impurity - N_t_R / N_t * right_impurity
                        - N_t_L / N_t * left_impurity)

    min_impurity_split : float, default=1e-7
        停止增长的阈值,小于这个值直接返回.

    Attributes
    ----------
    classes_ : array of shape (n_classes,) or a list of such arrays
        表示所有的类

    feature_importances_ : ndarray of shape (n_features,)
        特征重要性, 被选择最优特征的次数,进行降序.

    tree_ : Tree object
        The underlying Tree object.
    """
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_impurity_decrease=0.,
                 min_impurity_split=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.classes_ = None
        self.max_features_ = None
        self.decision_tree = None
        self.all_feats = None

    def fit(self, X, y):
        """使用X和y训练决策树的分类模型

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
         The training input samples. Internally, it will be converted to
         ``dtype=np.float32``

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
         The target values (class labels) as integers or strings.

        check_input : bool, (default=True)
         Allow to bypass several input checking.

        Returns
        -------
        self : object
         Fitted estimator.
        """
        if isinstance(X, list):
            X = self._check_array(X)
        if isinstance(y, list):
            y = self._check_array(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError('输入的数据 X 和 y 长度不匹配')

        self.classes_ = list(set(y))

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        data_origin = np.c_[X, y]
        self.all_feats = [i for i in range(X.shape[1])]
        self.max_features_ = X.shape[0]

        data = copy.deepcopy(data_origin)
        self.decision_tree = self._build_tree(data, 0)

    def _build_tree(self, data, depth):
        """创建决策树的主要代码

        Parameters:
        -----------
        data : {array-like} of shape (n_samples, n_features) + {label}
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32``

        depth: int, 树的深度

        Returns:
        -------
        DecisionTree

        """
        labels = np.unique(data[:, -1])
        # 只剩下唯一的类别时,停止,返回对应类别
        if len(labels) == 1:
            return DecisionTree(result=list(labels)[0])

        # 遍历完所有特征时,只剩下label标签,就返回出现字数最多的类标签
        if not self.all_feats:
            return DecisionTree(result=np.argmax(np.bincount(data[:, -1].astype(int))))

        # 超过最大深度,则停止,使用出现最多的参数作为该叶子节点的类
        if self.max_depth and depth > self.max_depth:
            return DecisionTree(result=np.argmax(np.bincount(data[:, -1].astype(int))))

        # 如果剩余的样本数大于等于给定的参数 min_samples_split,
        # 则不在进行分割, 直接返回类别中最多的类,该节点作为叶子节点
        if self.min_samples_split >= data.shape[0]:
            return DecisionTree(result=np.argmax(np.bincount(data[:, -1].astype(int))))

        # 叶子节点个数小于指定参数就进行返回,叶子节点中的出现最多的类
        if self.min_samples_leaf >= data.shape[0]:
            return DecisionTree(result=np.argmax(np.bincount(data[:, -1].astype(int))))

        # 根据基尼指数选择每个分割的最优特征
        best_idx, best_val, min_gini = self._get_best_feature(data)
        print("Current best Feature:", best_idx, best_val, min_gini)
        # 如果当前的gini指数小于指定阈值,直接返回
        if min_gini < self.min_impurity_split:
            return DecisionTree(result=np.argmax(np.bincount(data[:, -1].astype(int))))

        leftData, rightData = self._split_data(data, best_idx, best_val)

        # ============================= show me your code =======================
        # here
        leftDecisionTree = self._build_tree(leftData, depth+1)
        rightDecisionTree = self._build_tree(rightData, depth+1)
        # ============================= show me your code =======================

        return DecisionTree(col=best_idx, val=best_val, left_child=leftDecisionTree, right_child=rightDecisionTree)

    def _get_best_feature(self, data):
        """得到最优特征对应的列
        Parameters:
        ---------
        data: np.ndarray
            从data中选择最优特征

        Returns:
        -------
        bestInx, val, 最优特征的列的索引和使用的值.
        """
        best_idx = -1
        best_val = None
        min_gini = 1.0
        # 遍历现在可以使用的特征列
        # ============================= show me your code =======================
        # here
        currentGain = self.gini(data)
        rows_length = len(data)

        for col in self.all_feats:

            col_value_set = set([x[col] for x in data])
            for value in col_value_set:

                left, right = self._split_data(data, col, value)
                p = len(left) / rows_length
                gain = currentGain - p * self.gini(left) - (1 - p) * self.gini(right)

                if gain > min_gini:
                    min_gini = gain
                    best_idx = col
                    best_val = value

        # ============================= show me your code =======================
        # 删除使用过的特征
        self.all_feats.remove(best_idx)

        return best_idx, best_val, min_gini

    def gini(self, labels):
        """计算基尼指数.

        Paramters:
        ----------
        labels: list or np.ndarray, 数据对应的类目集合.

        Returns:
        -------
        gini : float ``` Gini(p) = \sum_{k=1}^{K}p_k(1-p_k)=1-\sum_{k=1}^{K}p_k^2 ```

        """
        # ============================= show me your code =======================

        # here
        length = len(labels)

        results = {}
        for label in labels:
            try:
                results[label[-1]] += 1
            except KeyError:
                results[label[-1]] = 1

        imp = 0.0
        for i in results:
            imp += results[i] / length * results[i] / length
        gini = 1 - imp

        # ============================= show me your code =======================
        return gini

    def _split_data(self, data, feat_col, val):
        '''根据特征划分数据集分成左右两部分.
          Parameters:
          ---------
          data:
            np.ndarray, 分割的数据

          feat_col:
            int, 使用第几列的数据进行分割

          val: int or float or str, 分割的值
              int or float : 使用比较方式
              str : 使用相等方式

          Returns:
          -------
          leftData, RightData
              int or left: leftData <= val < rightData
              str : leftData = val and rightData != val
          '''
        if isinstance(val, str):
            leftData = data[data[:, feat_col] == val]
            rightData = data[data[:, feat_col] != val]
        elif isinstance(val, (int, float)):
            leftData = data[data[:, feat_col] <= val]
            rightData = data[data[:, feat_col] > val]
        else:
            raise ValueError('except one of the [str, int, float]')
        return leftData, rightData

    def _check_array(self, X):
        """检查数据类型
        Parameters:
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        X: {array-like} of shape (n_samples, n_features)
        """
        if isinstance(X, list):
            X = np.array(X)
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("输出数据不合法, 目前只支持 np.ndarray or pd.DataFrame")
        return X


if __name__ == '__main__':

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    if __name__ == "__main__":
        # 分类树
        X, y = load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf = DecisionTreeClassifier()

        clf.fit(X_train, y_train)

        print("Classifier Score:", clf.score(X_test, y_test))
