# _*_coding:utf-8 _*_
import utils.FormatUtil as fu
import utils.WriterUtil as wu
import fasttext
from utils.PathUtil import Path

p = Path()
dataDirectoryPath = p.data_directory
oriDataPath = p.ori_data
clsPath = dataDirectoryPath + "/classes.txt"

formatedTrainPath = dataDirectoryPath + "/train.txt"
formatedTestPath = dataDirectoryPath + "/test.txt"
resultRecordPath = Path.join(dataDirectoryPath,"record_result.txt")
modelPath = Path.join(Path.join(p.root, 'model'), 'fasttext.model')


wu.export_classes(fu.filter_out_classes(oriDataPath, 1), clsPath)
fu.transfer_to_ft_format(oriDataPath, dataDirectoryPath, clsPath)

# 训练模型
print('--------------------')
print('训练模型中....')
dim = 50
epoch = 40
word_ngrams = 1
lr = 0.1
lr_update_rate = 150
loss = 'softmax'     # ns,hs,softmax
classifier = fasttext.supervised(formatedTrainPath, modelPath,
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

result = classifier.test(formatedTestPath)
print(result.precision)
print(result.recall)
with open(resultRecordPath, 'a', encoding='utf-8') as f:
    f.write("dim:"+str(dim) +
            " epoch:"+str(epoch) +
            " lr:"+str(lr) +
            " word_ngrams:"+str(word_ngrams) +
            " lr_update_rate:"+str(lr_update_rate) +
            " loss:"+loss +
            "   score:"+str(result.precision)+"\n")
