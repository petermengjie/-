#import jieba
import os
import pkuseg
import re
#jieba.load_userdict('dict.txt')


class MappingLable():
    def __init__(self,
                 lable_path='sign_lable.txt',
                 dict_txt='dict.txt',
                 image_dir=r'C:\Users\guoxiansheng\Desktop\Documents\Pictures of Signs'
                 ):
        self.lable_path = lable_path
        self.image_path = image_dir
        self.dict_txt = dict_txt
        self.lable = {}
#        self.lable_list = []
#        self.img_path_list = []
        self.get_lable_dict()
        self.pku=pkuseg.pkuseg()
    def get_lable_dict(self):
        with open('sign_lable.txt', 'r+', encoding='utf-8') as fp:
            for lin in fp:
                key, value = lin.split()
                self.lable[key] = value
#                self.lable_list.append(key)
    def chinese_to_img(self,string):
#        self.img_path_list = []
        punctuation=r'.,?!！。？，'
        string=re.sub(r"[%s]+" %punctuation, " ",string)
        mytext = self.pku.cut(string)
        for word in mytext:
            if word in self.lable:
                img_path = os.path.join(self.image_path, self.lable[word] + '.png')
            else:
                img_path=os.path.join(self.image_path, '4414.png')
            yield word,img_path


# if __name__ == '__main__':
#     test = MappingLable()
#     test.chinese_to_img('我今天很高兴')