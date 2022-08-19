from aip import AipSpeech
import time
import os
# """ 你的 APPID AK SK """
# APP_ID = '24245330'
# API_KEY = 'hKKvufVHjMH34TAecAMIBOVE'
# SECRET_KEY = 'nvHzcP22RPgdOqTmseSclhX6XO0ZzTp5'
#
# client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
#
# string = '你好百度, 壭壱売壳壴壵壶壷壸壶'
# result = client.synthesis(string, 'zh', 1, {
#     'vol': 5, 'per': 4
# })
#
# # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
# if not isinstance(result, dict):
#     with open('auido.mp3', 'wb') as f:
#         f.write(result)


class ApiAuido():
    def __init__(self):
        self.APP_ID = '24245330'
        self.API_KEY = 'hKKvufVHjMH34TAecAMIBOVE'
        self.SECRET_KEY = 'nvHzcP22RPgdOqTmseSclhX6XO0ZzTp5'
        self.client = AipSpeech(self.APP_ID, self.API_KEY, self.SECRET_KEY)
        self.initial()

    @staticmethod
    def initial():
        os.makedirs('./audio',exist_ok=True)
        for i in filter(lambda f: f.endswith('.mp3'), os.listdir('./audio')):
            os.remove(os.path.join('audio', i))

    def get_auido(self,string):
        name = 'audio/'+str(time.time())+'.mp3'
        result = self.client.synthesis(string, 'zh', 1, {
            'vol': 5, 'per': 4
        })
        if not isinstance(result, dict):
            with open(name, 'wb') as f:
                f.write(result)
            return name

if __name__ == '__main__':
    a = ApiAuido()
    a.get_auido('你好')