from torch.utils.data import DataLoader

class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # 到达末尾，重新初始化迭代器
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        return batch

    def __iter__(self):
        return self