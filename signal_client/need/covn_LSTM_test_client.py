import torchvision.transforms as transforms
import pickle
import os
import requests
import json
from PIL import Image
import torch

sample_size=128
sample_duration=16
#url='http://127.0.0.1:8080'


def read_images(folder_path,url='http://222.24.63.155:8080',headers={'headers:':'default'}):
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    assert len(os.listdir(folder_path)) >= sample_duration, "Too few images in your data folder: " + str(folder_path)
    images = []
    start = 1
    step = int(len(os.listdir(folder_path))/sample_duration)
    for i in range(sample_duration):
        x = os.path.join(folder_path, '{:06d}.jpg'.format(start+i*step))
        # x = os.path.join(folder_path, '{:06d}.jpg')
        #x = folder_path + '\\' + '{:06d}.jpg'
        #image = Image.open(x.format(start+i*step))  #.convert('L')
        image = Image.open(x)  #.convert('L')
        if transform is not None:
            image = transform(image)
        images.append(image)
    images = torch.stack(images, dim=0)
    res=requests.post(url+'/excute',headers=headers,data=pickle.dumps(images),timeout=15).text
    result=json.loads(res)
    if result['code']==0:
        return result['data']
