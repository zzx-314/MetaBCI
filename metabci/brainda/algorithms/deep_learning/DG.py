import numpy as np

class kFoldGenerator():
    '''
    数据生成器
    '''
    k = -1           # 折数
    x_list = []      # 包含k个元素的x列表
    y_list = []      # 包含k个元素的y列表
    fold_size = None  # 每份数据包含的人数，可调参数

    # 初始化
    def __init__(self, x, y, fold_size):
        if len(x) != len(y):
            assert False, '数据生成器：x或y的长度不等于k。'
        self.k = len(x)
        self.x_list = x
        self.y_list = y
        self.fold_size = fold_size

    def getFold(self, i):
        start_index = i * self.fold_size
        end_index = min((i + 1) * self.fold_size, len(self.x_list))
        val_indices = range(start_index, end_index)

        # 创建训练集索引
        train_indices = [idx for idx in range(len(self.x_list)) if idx not in val_indices]

        isFirst = True
        for p in train_indices:
            if isFirst:
                train_data = self.x_list[p]
                train_targets = self.y_list[p]
                isFirst = False
            else:
                train_data = np.concatenate((train_data, self.x_list[p]))
                train_targets = np.concatenate((train_targets, self.y_list[p]))

        val_data = np.concatenate([self.x_list[p] for p in val_indices])
        val_targets = np.concatenate([self.y_list[p] for p in val_indices])

        return train_data, train_targets, val_data, val_targets


