---
title: 炼丹中缺失的一课2
mathjax: true
date: 2023-09-06 09:45:53
tags: [深度学习]
categories: [深度学习]
---

​	筛选丹方上炉开炼的过程是少不了dataset和dataloader的，只不过我在早期炼丹的时候，由于那时候需要更加关心怎么出效果，怎么赶紧出活，没怎么在乎这个功能是怎么实现的。

<!--more-->

​	曾经有无数个日与夜，我都是这样：

```python
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

data = torch.randn(100, 64)
labels = torch.randint(0, 2, (100,))

custom_dataset = CustomDataset(data, labels)

batch_size = 64
shuffle = True
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)

for batch_data, batch_labels in dataloader:
    ...

```

​	我好像从来没有在乎过，`__getitem__`的输入idx是怎么来的，以及这个DataLoader是怎么凑够batchsize个样本的。只知道自己继承dataset写一个类，然后覆写它的几个魔法方法，然后再用torch里的dataloader封装一下，我就可以遍历数据集了。

​	这在大部分简单的任务里都行得通，但为了以后能掌控更加复杂的任务，以及看懂别人的代码，我们最好理解一下dataloader的实现。

​	实际上，在torch中，sampler和dataset是dataloader的两个部分。sampler用来生成索引，dataset是根据索引去读数据（图片，标签，序列等）。同时，我们知道，如果Python中的一个对象能被for...in...遍历，那它需要有`__iter__`,`__next__`方法，无论是其本身直接实现`__next__`，还是内部间接返回（使用一些高级语法`__getitem__`,`yield`，或者返回一个有`__next__`的类）。

​	所以我们现在从for...in...来观察一个整个过程，for...in...操作其实会被Python解释为：

```python
A = iter(A)
while True:
    try:
        value = next(A)
    except StopIteration:
        break
```

​	所以如果我们打开torch中dataloader.py的实现，我们会找到其`__iter__`方法：

```python
    def __iter__(self) -> '_BaseDataLoaderIter':
        # When using a single worker the returned iterator should be
        # created everytime to avoid reseting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()
```

​	我们可以看到，函数注解->提示我们这个方法会返回一个`_BaseDataLoaderIter`，我们跟着`self._get_iterator()`往下找：

```python
    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)
```

​	我们可以发现这里是来确定单进程还是多进程的，我们这里就只分析单进程了。所以我们转入`_SingleProcessDataLoaderIter(self)`。

​	注意，此时我们还都在DataLoader的类定义下，`_SingleProcessDataLoaderIter(self)`是将DataLoader所有的对象作为参数传入了`_SingleProcessDataLoaderIter()`的构造函数`__init__`。而这个类其实是继承自`_BaseDataLoaderIter()`的，所以这就是为什么函数注解告诉我们返回的是`_BaseDataLoaderIter(object)`。

```python
class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data
```

​	这个类也正是for...in...执行后，第一行的`A=iter(A)`得到的返回值，下面我们会关心`_SingleProcessDataLoaderIter`是否有`__next__`方法，其本身没有定义`__next__`，所以我们去找一下它的父类`_BaseDataLoaderIter`。在其定义中我们可以看到：

```python
    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                self._reset()
            data = self._next_data()
            self._num_yielded += 1
            if self._dataset_kind == _DatasetKind.Iterable and \
                    self._IterableDataset_len_called is not None and \
                    self._num_yielded > self._IterableDataset_len_called:
                warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                            "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                  self._num_yielded)
                if self._num_workers > 0:
                    warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                 "IterableDataset replica at each worker. Please see "
                                 "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
                warnings.warn(warn_msg)
            return data
```

​	所以`_SingleProcessDataLoaderIter`继承自`_BaseDataLoaderIter`的`__next__`，同时覆写了其中用到的`_next_data()`，覆写的`_next_data()`复用了父类的`_next_index()`，其定义很简短：

```python
    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration
```

​	所以此时，我们大概勾勒出了运行时的模样，当我们for...in...一个dataloader时，首先会先给出一个迭代器`_SingleProcessDataLoaderIter`。然后这个迭代器不断的调用next，调用时，`_next_data`内部的`_next_index`负责给出一个索引，然后会去抓取这个索引对应的数据。

​	我们已经有了不少的进展了，下面要继续看下去。

​	首先我们要关注`_next_index`里的`self._sampler_iter`，这个对象是在父类`_BaseDataLoaderIter`初始化时产生的，与其相关的两行代码为：

```python
        self._index_sampler = loader._index_sampler
		...
        self._sampler_iter = iter(self._index_sampler)
```

​	所以我们要进一步寻找这个`self._index_sampler`的`__iter__`方法，来搞清楚这个（至少是可迭代的）对象是怎么来的。我们不难发现，其定义是在DataLoader中的：

```python
    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler
```

​	而至于`self.batch_sampler`和`self.sampler`，其在DataLoader的初始化中就设定了：

```python
        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    # Cannot statically verify that dataset is Sized
                    # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
                    sampler = RandomSampler(dataset, generator=generator)  # type: ignore
                else:
                    sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator
```

​	同时，上面这段代码也是为什么我们一般无需手动在给一个dataloader时，输入sampler的原因。我们一般不会给出sampler，更不会给出batch_sampler。我们一般都是给出batchsize和shuffle，根据上面的代码我们可以知道如果我们只给出batchsize和shuffle，会自动的将一个RandomSampler赋给self.sampler，然后将一个BatchSampler赋给self.batch_sampler。

​	例如在最简单的情形下的`SequentailSampler`：

```python
class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)
```

​	此时，`iter(self._index_sampler)`就会返回`iter(range(len(self.data_source)))`这个迭代器，所以`next()`的结果就是0,1,2,...。

> 在Python中，range()本身并不是迭代器，它仅仅是一个可迭代对象，但iter(range)会自动返回一个迭代器。这个设计使得range()听起来很奇怪，但其实背后是有更深层的原因的。

​	现在我们明确了，当实例化一个dataloader时，会根据输入参数得到类内成员sampler。然后sampler会向下穿越到`_SingleProcessDataLoaderIter`，来完成“得到索引”的使命。所以到这里，dataloader的两部分，sampler和dataset，我们已经解决其中之一了。

​	现在距离谜团彻底解开，还差`_SingleProcessDataLoaderIter`里的那一行：

```python
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)
```

​	这其实就是dataset的那一部分，我们跳转到`_DatasetKind`：

```python
class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
```

​	我们发现它好像约定了两种dataset，Map和Iterable。令人震惊的是，大多数人（包括我）可能炼丹到现在都不一定用过，甚至不知道Iterable这种形式的数据集。（但其实这两种数据集也没那么大差异罢了）

​	如果你像文章最开头那样，每次都用torch.utils.data.Dataset来覆写的话，那你使用的就是标准的Map式数据集。这种方式是使用`__getitem__`和`__len__`来达到对数据集的随机访问（需要`__len__`来保证访问是否越界），一般而言，这种设计方式鼓励的是在`__init__`时将数据集整个都存入内存中，这样就可以快速的进行访问，减少IO时间。

> 但其实也并不用严格的将整个数据集都load进内存里，完全可以`_init`时候只给定路径，然后每`__getitem__`的时候现读。这又何尝不是一种Iterable？

​	而Iterable的数据集，可能是在更加实际的场景下用的。它继承的是torch.utils.data.IterableDataset，每次只需要重写`__iter__`就好啦：

```python
from torch.utils.data import IterableDataset

class IterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
         ...
         yield item
```

​	由于其`__iter__`会返回一个生成器，所以需要先将数据集iter()一下返回一个迭代器，这里由于生成器机制yield可以next()，所以如果读数据就疯狂next()就好了。

​	所以说回来，我们现在只需要看一下`_utils.fetch`是怎么实现的就好了：

```python
class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
```

​	可以看到，和我们之前的分析是一致的，如果是Iterable的数据集，就会`self.dataset_iter = iter(dataset)`以后然后next，如果是Map的数据集，那就会直接用[]来访问。

​	最后的`self.collate_fn`是一个用于“整理”的函数，如果去查阅其源码，其意思基本就是把各种形式的数据集强制转换为tensor。我们一般都不会用到这个功能，或许在自然语言处理里，用于处理长短不一的输入时，会重写一个自己的`collate_fn`。

​	所以，我们的一个batch的数据是怎么来的呢？现在我们可以有了圆满的答案，注意在dataloader的类定义中，有一个函数：

```python
    @property
    def _auto_collation(self):
        return self.batch_sampler is not None
```

​	而这个方法第一次被调用，是在dataloader的`__init__`中：

```python
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert
```

​	如果我们设置了batchsize，那么此时的`self.batch_sampler`就不为空，`self._auto_collation`就为真。所以在后面的`_indexIsampler`里，就会自动使用`self.batch_sampler`，但其实BatchSampler只是对sampler的又一层封装：

```python
class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore
```

​	我们可以看到，BatchSampler里的`__iter__`，忠实的完成了凑够一个batchsize的数据的职责。

​	所以，最后的递归调用可以总结成：
$$
\mathrm{for}\ \mathrm{data}, \mathrm{label}\ \mathrm{in}\ \mathrm{loader}:
\\
\downarrow 
\\
\mathrm{DataLoader}.\_\_\mathrm{iter}\_\_()
\\
\downarrow 
\\
\mathrm{DataLoader}.\_\_\mathrm{get}\_\mathrm{iterator}\_\_()
\\
\downarrow 
\\
\_\mathrm{SingleProcessDataLoader()}
\\
\downarrow 
\\
\_\mathrm{BaseDataLoaderIter}.\_\_\mathrm{next}\_\_()
\\
\downarrow 
\\
\_\mathrm{SingleProcessDataLoader}.\_\mathrm{next}\_\mathrm{data()}
\\
\downarrow 
\\
\_\mathrm{BaseDataLoaderIter}.\_\mathrm{next}\_\mathrm{index()}
\\
\downarrow 
\\
\_\mathrm{SingleProcessDataLoader}.\_\mathrm{dataset}\_\mathrm{fetcher()}
\\
\downarrow 
\\
\mathrm{return}\ \mathrm{a}\ \mathrm{batch}
$$
​	多进程时的情况这里就不作分析了，涉及到一系列进程守护，队列之类的机制。毕竟写这篇blog只是为了理清另一个代码库里的几行代码罢了，总之，这就是作为dataloader，传奇的运行周期，它尽职尽责的将数据load进来，为纷至沓来的炼丹师编织一个又一个美好的梦境。

### End

​	沉淀！

<center>
    <img src='/images/dataloader/dataloader_1.jpg'  style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>
