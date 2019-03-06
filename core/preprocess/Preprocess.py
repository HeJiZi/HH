import utils.ReadUtil as ru

from utils.PathUtil import Path
import utils.CutExtend as cutEx
import core.preprocess.DataFrequence as df

class Preprocess:
    df_dict={}
    path = Path()

    def calculate_word_cout(self):
        corpus = []
        data = ru.get_item_names(self.path.ori_data)
        LEN = len(data)
        index = 0
        for item in data:
            index += 1
            if index % 1024 == 0:
                print("正在切分第" + str(index) + "数据\n")
            item = cutEx.seg_depart(item)
            text = " ".join(str(i) for i in item.split(" "))
            corpus.append(text)

        print("开始计算DF\n")

        index = 0
        IDFdict = df.my_df(corpus)
        IDFdict = df.sort_dict(IDFdict)

        fp = open(self.path.count, 'w', encoding="utf-8")
        for j in IDFdict:
            fp.write(j + ' ' + str(IDFdict[j]) + '\n')
        fp.close()
        print("完成")

        self.df_dict = IDFdict
        return IDFdict


    def init_dict(self,have_count_file = False):
        df = {}
        if(have_count_file):
            with open(self.path.count, 'r', encoding="utf-8") as count_file:
                for line in count_file:
                    t = line.split(' ')
                    df[t[0]] = t[1].strip() # 去除行尾的换行符
        else:
            df = self.calculate_word_cout()

        custom_dict_file = open(self.path.custom_dict, 'w', encoding="utf-8")
        with open(self.path.dict, 'r', encoding="utf-8") as dict_file:
            for line in dict_file:
                t = line.split(' ')
                if(t[0].strip().isspace()):
                    print("出现空白符",t[0])
                if(t[0].strip() in df.keys()):
                    custom_dict_file.write(line)



# def get_vectors(init = True):
#     if(init):
#         preprocess = Preprocess()
#         preprocess.calculate_word_cout()


# Preprocess().init_dict(have_count_file=True)
Preprocess().calculate_word_cout()