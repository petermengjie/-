# 手语识别系统

基于pytorch深度学习框架，设计并实现了序列到序列的算法seq2seq，用于对长句的手语进行自动识别；还设计并实现了时序数据处理算法convlstm，用于对手语单词进行识别。两个深度学习算法利用国内开源手语数据集DEVISIGN数据集进行训练，能够得到一个目前比较理想的准确度。整个系统界面采用了qt5进行开发，并内置了手语词典、语音朗读等功能。

已经实现的功能：

- 连续手语动作识别（翻译成句子）
- 单个手语动作识别（翻译成单词）
- 摄像头采集视频
- 导入现有手语视频
- 手语词典，能够对手语表达含义到手语动作进行一个逆向翻译
- 发音功能，利用百度api对识别的结果进行发音

手语识别系统共开发了两个版本，第一个版本为离线版本，支持在本地单机进行手语的识别，模型加载均在本地进行。第二个版本为client-server版本，因为有的客户端性能较差，手语模型需要消耗比较高的资源，模型加载缓慢或者性能不足报错，因此改进设计了整个版本。

## 手语识别系统单机离线版

### 系统运行

```
cd signal_loca
python qt5.py
```

### 运行界面

<img src="https://raw.githubusercontent.com/petermengjie/-/master/imgs/image-20220819154708237.png" alt="image-20220819154708237" style="zoom:80%;" />

打开摄像头按钮能够利用摄像头对手语动作进行采集，打开视频文件能够对导入的手语视频文件进行识别。

### 数据载入模型

点击载入数据按钮，将手语数据载入模型，等待模型输出结果，结果会直接显示在系统中。

<img src="https://raw.githubusercontent.com/petermengjie/-/master/imgs/image-20220819154852626.png" alt="image-20220819154852626" style="zoom:80%;" />

### 切换模型

左下角按钮，Net1为连续手语识别模型，Net2为手语词汇识别模型，可以在任意两个模型直接进行切换。

![image-20220819155110613](https://raw.githubusercontent.com/petermengjie/-/master/imgs/image-20220819155110613.png)

### 发音功能

点击左下角喇叭按钮，即可对识别的结果进行自动发音。注意先在signal_local/need/baiduAPI2mp3.py中配置好百度api中的API_KEY之类的信息。

### 手语翻译

利用分词模型jieba对句子进行分词，并在手语词典中进行反向查询，在系统界面可显示翻译结果。

<img src="https://raw.githubusercontent.com/petermengjie/-/master/imgs/image-20220819155510941.png" alt="image-20220819155510941" style="zoom:80%;" />

## 手语识别系统CS版

为了避免本地机器的性能不足导致的识别速度缓慢等问题，开发了第二个版本。功能和离线版本一致，手语视频的处理等过程均在本地机器上进行，处理好的数据发送给后端服务器进行手语识别。服务器后端采用了python轻量型框架web.py进行开发，具体实现见代码signal_service/code.py。前端代码见signal_client/qt5.py。

### 服务端接口

服务端共设计了3个接口，如下图所示。

![image-20220819160248015](https://raw.githubusercontent.com/petermengjie/-/master/imgs/image-20220819160248015.png)

不同请求uri代表着不同的含义，/getid代表返回一个随机的id，能够对不同客户端的身份进行设别，避免在模型切换过程中出现混乱。/set能够在不同模型直接进行快速转换，/excute能够对上传的数据进行识别，并返回识别结果。

数据返回格式均使用了json格式，json格式体积较小并且解析方便，能够方便开发。

### 运行服务端

```
cd signal_service
python code.py
```

服务端默认监听所有网卡，端口为8080。

### 运行客户端

```
cd signal_client
python qt5.py
```

可在qt5.py文件中的CamaraPageWindow类中设置url属性，即可使用你想要的服务端。

<img src="https://raw.githubusercontent.com/petermengjie/-/master/imgs/image-20220819160914183.png" alt="image-20220819160914183" style="zoom:80%;" />

客户端界面和离线版本保持一致，

点击切换模型等按钮，即可在服务端后台看到客户端的不同请求日志。

<img src="https://raw.githubusercontent.com/petermengjie/-/master/imgs/image-20220819161212959.png" alt="image-20220819161212959" style="zoom:80%;" />