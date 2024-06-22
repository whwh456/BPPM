"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset

#这段代码通过构造数据集模块的文件名，动态导入该模块，然后在该模块（包）中寻找与给定数据集名称相关的类。
# 找到符合条件的类后，将其返回,用于之后实例化。如果没有找到匹配的类，会抛出 NotImplementedError 异常。
def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)    # 引入包模块

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

# 返回的dataset有dataloader对象属性
def create_dataset(opt,test_batch_size=None):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        # >>> from data import create_dataset
        # >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt,test_batch_size)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt,test_batch_size=None):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        # dataset_mode需要是创建的dataset类的名称，比如示例中的"align"
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        # find_dataset_using_name 通过构造数据集模块的文件名，动态导入该模块，然后在该模块中寻找与给定数据集名称相关的类并返回。
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created_%s" % (type(self.dataset).__name__,opt.datasettype))
        batch_size=opt.batch_size
        if test_batch_size!=None:
            batch_size=test_batch_size
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=opt.num_worker,
            shuffle=opt.shuffle)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data


def getcityscaperoot(opt,location):
    opt.dataset_mode="cityscape"
    if location == 0:
        opt.dataroot = "/home/liujian/chenlqn/mycode/cityscape"
    elif location == 1:
        opt.dataroot = "/haowang_ms/chenlqn/mycode/cityscape"
    elif location == 2:
        opt.dataroot = "F:\mycode\cityscape"
    else:
        raise NotImplementedError('location [%s] is valid is not found')