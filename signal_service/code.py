import web
from seq2seq_test import SignLanguagePredict
import os
import json
from covn_LSTM_test import SignLanguage_Isolate_Predict
import string
import random

urls = (
    '/excute', 'index',
    '/set', 'change',
    '/getid', 'genid'
)
img_dir='images'
if not os.path.exists(img_dir):
    os.mkdir(img_dir)
net_id=(0,1)
net_list=[]
net1=SignLanguagePredict()
net_list.append(net1)
net2=SignLanguage_Isolate_Predict()
net_list.append(net2)
in_data_list=["images","images2"]
in_data=in_data_list[0]
sessions={}
sessions['default']=0

class index:
    def GET(self):
        web.header('Content-Type', 'text/html;charset=UTF-8')
        try:
            session=web.ctx.env.get('HTTP_SESSION')
            if not session:
                session='default'
            elif session not in sessions:
                sessions[session]=0
            in_data=in_data_list[sessions[session]]
            net=net_list[sessions[session]]
            net.read_images(os.path.join(img_dir,session,in_data))
            net.test()
            return net.out_word
        except:
            return json.dumps({'code':400,'data':'unknow error'})
    def POST(self):
        try:
            session=web.ctx.env.get('HTTP_SESSION')
            if not session:
                session='default'
            elif session not in sessions:
                sessions[session]=0
            if not os.path.exists(os.path.join(img_dir,session)):
                os.mkdir(os.path.join(img_dir,session))
            in_data=in_data_list[sessions[session]]
            net=net_list[sessions[session]]
            with open(os.path.join(img_dir,session,in_data),'wb') as f:
                content=web.data()
                f.write(content)
            net.read_images(os.path.join(img_dir,session,in_data))
            net.test()
            return json.dumps({'code':0,'data':net.out_word})
            #return json.dumps({'code':0,'data':'success'})
        except:
            return json.dumps({'code':400,'data':'unknow error'})

class change:   #更换网络模型
    def GET(self):
        try:
            session=web.ctx.env.get('HTTP_SESSION')
            if not session:
                session='default'
            data=web.input()
            if len(data) == 0:
                return json.dumps({'code':400,'data':'input none'})
            result=int(data.net)
            if result in net_id:
                sessions[session]=result
                return json.dumps({'code':0,'data':'success'})
            else:
                return json.dumps({'code':400,'data':'input error'})
        except:
            return json.dumps({'code':400,'data':'unknow error'})

class genid:    #返回随机Id,用于识别不同客户端==session
    def GET(self):
        return ''.join(random.sample(string.ascii_letters + string.digits, 12))


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
