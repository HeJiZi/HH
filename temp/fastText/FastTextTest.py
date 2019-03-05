# _*_coding:utf-8 _*_
import utils.FormatUtil as fu
import utils.WriterUtil as wu
import fasttext

dataDirectoryPath = "/data"
oriDataPath = dataDirectoryPath + "/train.tsv"
clsPath = dataDirectoryPath + "/classes.txt"

formatedTrainPath = dataDirectoryPath + "/train.txt"
formatedTestPath = dataDirectoryPath + "/test.txt"


wu.export_classes(fu.filter_out_classes("./data/train.tsv", 1), './data/classes.txt')
fu.transfer_to_ft_format('./data/train.tsv', './data', './data/classes.txt')

# 训练模型
print('--------------------')
print('训练模型中....')
dim = 50
epoch = 40
word_ngrams = 1
lr = 0.1
lr_update_rate = 150
loss = 'softmax'     # ns,hs,softmax
classifier = fasttext.supervised("./data/train.txt", "./model/fasttext.model",
                                 label_prefix="_label_",
                                 lr=lr,
                                 dim=dim,
                                 epoch=epoch,
                                 word_ngrams=word_ngrams,
                                 bucket=2000000,
                                 lr_update_rate=lr_update_rate,
                                 loss=loss)   # bucket=2000000
print('训练完毕')
print('--------------------')
# load训练好的模型
# classifier = fasttext.load_model('./model/fasttext.model.bin', label_prefix='_label_')

result = classifier.test("./data/test.txt")
print(result.precision)
print(result.recall)
with open('./data/record_result.txt', 'a', encoding='utf-8') as f:
    f.write("dim:"+str(dim) +
            " epoch:"+str(epoch) +
            " lr:"+str(lr) +
            " word_ngrams:"+str(word_ngrams) +
            " lr_update_rate:"+str(lr_update_rate) +
            " loss:"+loss +
            "   score:"+str(result.precision)+"\n")
