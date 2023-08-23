# mytransformer
本项目基本信息如下：

解决的问题：机器翻译，将中文翻译成英文（由于是demo版，翻译效果有限）。

主要用到的工具：pytorch 2.0.0+cu117（用于深度学习相关操作），spacy（用于中文/英文语法分词）。

数据来源：网上随便找的一段中-英互译文本（没有经过任何预处理）。

data文件夹：训练数据（来源见上），中文和英文分开存储。

runs/loss_plot文件夹：模型运行时输出的一些优化器参数，可以忽略。

build_vocab.py：从源文本建立词汇表。

zh_vocab.pkl/en_vocab.pkl：建立好的词汇表，使用pickle命令存到该文件中。

dataset.py：从源文本建立token化的数据集。

utils.py：训练过程中用到的一些数据结构，比如存储模型、翻译例句。

model.py：Transformer模型。

train.py：训练流程。
