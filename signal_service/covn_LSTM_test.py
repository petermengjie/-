import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import pickle




number_gpu = 0
torch.cuda.set_device(number_gpu)
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






class CRNN(nn.Module):
    def __init__(self, sample_size=256, sample_duration=16, num_classes=100,
                lstm_hidden_size=512, lstm_num_layers=1):
        super(CRNN, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # network params
        self.ch1, self.ch2, self.ch3, self.ch4 = 64, 128, 256, 512
        self.k1, self.k2, self.k3, self.k4 = (7, 7), (3, 3), (3, 3), (3, 3)
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (1, 1), (1, 1), (1, 1)
        self.p1, self.p2, self.p3, self.p4 = (0, 0), (0, 0), (0, 0), (0, 0)
        self.d1, self.d2, self.d3, self.d4 = (1, 1), (1, 1), (1, 1), (1, 1)
        self.lstm_input_size = self.ch4
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # network architecture
        # in_channels=3 for rgb
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.p1, dilation=self.d1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch1, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.p2, dilation=self.d2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch2, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.p3, dilation=self.d3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch3, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.p4, dilation=self.d4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch4, out_channels=self.ch4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.num_classes)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # print(x.shape)
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # Conv
            out = self.conv1(x[:, :, t, :, :])
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print(cnn_embed_seq.shape)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        # use faster code paths
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)
        # MLP
        # out: (batch, seq, feature), choose the last time step
        out = self.fc1(out[:, -1, :])

        return out


class SignLanguage_Isolate_Predict():
    def __init__(self,sample_size = 128,
                        sample_duration = 16,
                        num_classes = 100,
                        lstm_hidden_size = 512,
                        lstm_num_layers = 1,
                        label_path="dictionary.txt",
                        pthfile = 'slr_convlstm_epoch068.pth',
                        device = 'cpu'
                 ):
        self.__sample_size = sample_size
        self.__sample_duration = sample_duration
        self.__num_classes = num_classes
        self.__lstm_hidden_size = lstm_hidden_size
        self.__lstm_num_layers = lstm_num_layers
        self.label_path = label_path
        self.pthfile = pthfile
        self.device = device
        self.load_model()
        self.lable_dict()
        self.out_word=''
        # print(self.labels)
    def load_model(self):
        self.model = CRNN(sample_size=self.__sample_size, sample_duration=self.__sample_duration, num_classes=self.__num_classes,
                     lstm_hidden_size=self.__lstm_hidden_size, lstm_num_layers=self.__lstm_num_layers).to(device)
        self.model.load_state_dict(torch.load(self.pthfile, map_location='cpu'))

    def read_images(self,data):
        # global transform
        #transform = transforms.Compose([transforms.Resize([self.__sample_size, self.__sample_size]),
        #                                transforms.ToTensor(),
        #                                transforms.Normalize(mean=[0.5], std=[0.5])])
        #assert len(os.listdir(folder_path)) >= self.__sample_duration, "Too few images in your data folder: " + str(folder_path)
        #self.images = []
        #start = 1
        #step = int(len(os.listdir(folder_path))/self.__sample_duration)
        #for i in range(self.__sample_duration):
            # x = os.path.join(folder_path, '{:06d}.jpg')
        #    x = folder_path + '\\' + '{:06d}.jpg'
        #    image = Image.open(x.format(start+i*step))  #.convert('L')
        #    if transform is not None:
        #        image = transform(image)
        #    self.images.append(image)
#
#        self.images = torch.stack(self.images, dim=0)
        self.images = pickle.load(open(data,'rb'))
        # switch dimension
        self.images = self.images.permute(1, 0, 2, 3).unsqueeze(0)
        # print(images.shape)
        # return images


    def lable_dict(self):
        self.labels = {}
        try:
            label_file = open(self.label_path, 'r', encoding='utf-8')
            for line in label_file.readlines():
                line = line.strip()
                line = line.split('\t')
                self.labels[line[0]] = line[1]
        except Exception as e:
            raise

    def label_to_word(self, label):
        if isinstance(label, torch.Tensor):
            return self.labels['{:06d}'.format(label.item())]
        elif isinstance(label, int):
            return self.labels['{:06d}'.format(label)]

    def test(self):
            self.model.eval()
            with torch.no_grad():
                # get the inputs and labels
                inputs = self.images.to(device)
                # forward
                outputs = self.model(inputs)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                # compute the loss
                self.prediction = torch.max(outputs, 1)[1]
                self.out_word = self.label_to_word(self.prediction)
                # print('*********************')
                # print('Prediction:', self.prediction)
                # print('Prediction:', self.label_to_word(self.prediction))
                # print('*********************')




if __name__ == '__main__':
    # Load data
    image_path = r"./P01_01_00_4"
    test = SignLanguage_Isolate_Predict()
    test.read_images(image_path)
    test.test()

