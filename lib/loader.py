from torch.utils import data
def load_array(data_arrays, batch_size, shuffle):  #@save
    """构造一个PyTorch数据迭代器"""

    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle ,drop_last=True)

