
from sklearn.metrics import accuracy_score
import torchvision.models as models
import random
import numpy as np
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms




# Path setting

# dict_path = "dictionary.txt"
# corpus_path = "corpus.txt"
image_path = "../video/VideoCapture_img_2021-06-09_11-38-42"
# pthfile = 'slr_seq2seq_epoch050.pth'


# Log to file & tensorboard writer



# Use specific gpus
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
number_gpu = 0
torch.cuda.set_device(number_gpu)
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# sample_size = 128
# sample_duration = 48
# enc_hid_dim = 512
# emb_dim = 256
# dec_hid_dim = 512
# dropout = 0.5


class Encoder(nn.Module):
    def __init__(self, lstm_hidden_size=512, arch="resnet18"):
        super(Encoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

        # network architecture
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=True)
        # delete the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.lstm = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
        )

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # with torch.no_grad():
            out = self.resnet(x[:, :, t, :, :])
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        # use faster code paths
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)

        # num_layers * num_directions = 1
        return out, (h_n.squeeze(0), c_n.squeeze(0))
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim+enc_hid_dim, dec_hid_dim)
        self.fc = nn.Linear(emb_dim+enc_hid_dim+dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, context):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)

        # input(batch_size): last prediction
        # hidden(batch_size, dec_hid_dim): decoder last hidden state
        # cell(batch_size, dec_hid_dim): decoder last cell state
        # context(batch_size, enc_hid_dim): context vector
        # print(input.shape, hidden.shape, cell.shape, context.shape)
        # expand dim to (1, batch_size)
        input = input.unsqueeze(0)

        # embedded(1, batch_size, emb_dim): embed last prediction word
        embedded = self.dropout(self.embedding(input))

        # rnn_input(1, batch_size, emb_dim+enc_hide_dim): concat embedded and context
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)

        # output(seq_len, batch, num_directions * hidden_size)
        # hidden(num_layers * num_directions, batch, hidden_size)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        # hidden(batch_size, dec_hid_dim)
        # cell(batch_size, dec_hid_dim)
        # embedded(1, batch_size, emb_dim)
        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        embedded = embedded.squeeze(0)

        # prediction
        prediction = self.fc(torch.cat((embedded, context, hidden), dim=1))

        return prediction, (hidden, cell)
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, imgs, target, teacher_forcing_ratio=0.5):
        # imgs: (batch_size, channels, T, H, W)
        # target: (batch_size, trg len)
        batch_size = imgs.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs(batch, seq_len, hidden_size): all hidden states of input sequence
        encoder_outputs, (hidden, cell) = self.encoder(imgs)

        # compute context vector
        context = encoder_outputs.mean(dim=1)

        # first input to the decoder is the <sos> tokens
        input = target[:,0]

        for t in range(1, trg_len):
            # decode
            output, (hidden, cell) = self.decoder(input, hidden, cell, context)

            # store prediction
            outputs[t] = output

            # decide whether to do teacher foring
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token
            top1 = output.argmax(1)

            # apply teacher forcing
            input = target[:,t] if teacher_force else top1

        return outputs

# def dictionnary(corpus_path):
#    # dictionary
#     dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
#     output_dim = 3
#     try:
#         dict_file = open(corpus_path, 'r',encoding='utf-8')
#         for line in dict_file.readlines():
#             line = line.strip().split()
#             sentence = line[1]
#             for char in sentence:
#                 if char not in dict:
#                     dict[char] = output_dim
#                     output_dim += 1
#     except Exception as e:
#         raise
#
#     # # corpus
#     # corpus = {}
#     # unknown = set()
#     # try:
#     #     corpus_file = open(corpus_path, 'r',encoding='utf-8')
#     #     for line in corpus_file.readlines():
#     #         line = line.strip().split()
#     #         sentence = line[1]
#     #         raw_sentence = (line[1]+'.')[:-1]
#     #         paired = [False for i in range(len(line[1]))]
#     #         # print(id(raw_sentence), id(line[1]), id(sentence))
#     #         # pair long words with higher priority
#     #         for token in sorted(dict, key=len, reverse=True):
#     #             index = raw_sentence.find(token)
#     #             # print(index, line[1])
#     #             if index != -1 and not paired[index]:
#     #                 line[1] = line[1].replace(token, " "+token+" ")
#     #                 # mark as paired
#     #                 for i in range(len(token)):
#     #                     paired[index+i] = True
#     #         # add sos
#     #         tokens = [dict['<sos>']]
#     #         for token in line[1].split():
#     #             if token in dict:
#     #                 tokens.append(dict[token])
#     #             else:
#     #                 unknown.add(token)
#     #         # add eos
#     #         tokens.append(dict['<eos>'])
#     #         corpus[line[0]] = tokens
#     # except Exception as e:
#     #     raise
#     # # add padding
#     # length = [len(tokens) for key, tokens in corpus.items()]
#     # max_length = max(length)
#     # # print(max(length))
#     # for key, tokens in corpus.items():
#     #     if len(tokens) < max_length:
#     #         tokens.extend([dict['<pad>']]*(max_length-len(tokens)))
#     return dict
# def read_images(folder_path,frames=64):
#     # global transform
#     transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.5], std=[0.5])])
#     assert len(os.listdir(folder_path)) >= frames, "Too few images in your data folder: " + str(folder_path)
#     images = []
#     start = 1
#     step = int(len(os.listdir(folder_path))/frames)
#     for i in range(frames):
#         # x = os.path.join(folder_path, '{:06d}.jpg')
#         x = folder_path + '\\' + '{:06d}.jpg'
#         image = Image.open(x.format(start+i*step))  #.convert('L')
#         if transform is not None:
#             image = transform(image)
#         images.append(image)
#
#     images = torch.stack(images, dim=0)
#     # switch dimension
#     images = images.permute(1, 0, 2, 3).unsqueeze(0)
#     # print(images.shape)
#     return images
# def remove_endzeros_list(put_list):
#     '''
#     Remove the zero at the end of the list
#     :param put_list:
#     :return:
#     '''
#     n_0 = 0
#     for k in put_list[::-1]:
#         if k == 0:
#             n_0 += 1
#         else:
#             break
#     return put_list[:-n_0-1]
# def test_seq2seq(model, imgs, device, wold_dict):
#     model.eval()
#     all_trg = []
#     all_pred = []
#
#     with torch.no_grad():
#         target = np.arange(12).reshape(1,12)
#         target = torch.LongTensor(target)
#         imgs = imgs.to(device)
#         target = target.to(device)
#         # print(imgs.shape)
#         # forward(no teacher forcing)
#         outputs = model(imgs, target, 0)
#
#         # target: (batch_size, trg len)
#         # outputs: (trg_len, batch_size, output_dim)
#         # skip sos
#         output_dim = outputs.shape[-1]
#         outputs = outputs[1:].view(-1, output_dim)
#         target = target.permute(1, 0)[1:].reshape(-1)
#
#
#         # compute the accuracy
#         prediction = torch.max(outputs, 1)[1]
#         target_label, prediction_label = target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy()
#         score = accuracy_score(target_label, prediction_label)
#         all_trg.extend(target)
#         all_pred.extend(prediction)
#         target_label = remove_endzeros_list(target_label)
#         prediction_label = remove_endzeros_list(prediction_label)
#         print('*************************state*******************')
#         print('target:',target_label)
#         print('prediction:',prediction_label)
#         print("score:",score)
#         print('target_wold:',' '.join([wold_dict[x] for x in target_label]))
#         print('prediction_wold:', ' '.join([wold_dict[x] for x in prediction_label]))
#         print('*************************end*********************')
# def load_model():
#     encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)
#     decoder = Decoder(output_dim=253, emb_dim=emb_dim, enc_hid_dim=enc_hid_dim,
#                       dec_hid_dim=dec_hid_dim, dropout=dropout).to(device)
#     model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)
#
#     model.load_state_dict(torch.load(pthfile, map_location='cuda:0'))
#     return model

# def test():
#     # Load data
#
#     a = time.time()
#     # Create Model
#     model = load_model()
#     # print(model)
#
#     dict_word = dictionnary(corpus_path)
#     # print(dict_word)
#     wold_dict = dict([val, key] for key, val in dict_word.items())
#     # print(wold_dict)
#
#     images = read_images(image_path)
#
#     # Validate the model
#     test_seq2seq(model, images, device, wold_dict)
#     print(time.time()-a)

class SignLanguagePredict():
    def __init__(self,sample_size = 128,
                        sample_duration = 48,
                        enc_hid_dim = 512,
                        emb_dim = 256,
                        dec_hid_dim = 512,
                        dropout = 0.5,
                        corpus_path="../corpus.txt",
                        pthfile = '../slr_seq2seq_epoch050.pth',
                        device = 'cpu'
                 ):
        self.__sample_size = sample_size
        self.__sample_duration = sample_duration
        self.__enc_hid_dim = enc_hid_dim
        self.__emb_dim = emb_dim
        self.__dec_hid_dim = dec_hid_dim
        self.__dropout = dropout
        self.corpus_path = corpus_path
        self.pthfile = pthfile
        self.device = device
        self.load_model()
        self.dictionnary()
        self.wold_dict = dict([val, key] for key, val in self.dict.items())
        self.out_word=''
    def load_model(self):
        encoder = Encoder(lstm_hidden_size=self.__enc_hid_dim, arch="resnet18").to(self.device)
        decoder = Decoder(output_dim=253, emb_dim=self.__emb_dim, enc_hid_dim=self.__enc_hid_dim,
                          dec_hid_dim=self.__dec_hid_dim, dropout=self.__dropout).to(self.device)
        self.model = Seq2Seq(encoder=encoder, decoder=decoder, device=self.device).to(self.device)
        self.model.load_state_dict(torch.load(self.pthfile, map_location='cpu'))

    def read_images(self,folder_path):
        # global transform
        transform = transforms.Compose([transforms.Resize([self.__sample_size, self.__sample_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])
        assert len(os.listdir(folder_path)) >= self.__sample_duration, "Too few images in your data folder: " + str(folder_path)
        self.images = []
        start = 1
        step = int(len(os.listdir(folder_path))/self.__sample_duration)
        for i in range(self.__sample_duration):
            # x = os.path.join(folder_path, '{:06d}.jpg')
            x = folder_path + '\\' + '{:06d}.jpg'
            image = Image.open(x.format(start+i*step))  #.convert('L')
            if transform is not None:
                image = transform(image)
            self.images.append(image)

        self.images = torch.stack(self.images, dim=0)
        # switch dimension
        self.images = self.images.permute(1, 0, 2, 3).unsqueeze(0)
        # print(images.shape)
        # return images

    def remove_endzeros_list(self,put_list):
        '''
        Remove the zero at the end of the list
        :param put_list:
        :return:
        '''
        n_0 = 0
        for k in put_list[::-1]:
            if k == 0:
                n_0 += 1
            else:
                break
        return put_list[:-n_0 - 1]

    def dictionnary(self):
        # dictionary
        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        output_dim = 3
        try:
            dict_file = open( self.corpus_path, 'r', encoding='utf-8')
            for line in dict_file.readlines():
                line = line.strip().split()
                sentence = line[1]
                for char in sentence:
                    if char not in self.dict:
                        self.dict[char] = output_dim
                        output_dim += 1
        except Exception as e:
            raise
        # # corpus
        # corpus = {}
        # unknown = set()
        # try:
        #     corpus_file = open( self.corpus_path, 'r', encoding='utf-8')
        #     for line in corpus_file.readlines():
        #         line = line.strip().split()
        #         sentence = line[1]
        #         raw_sentence = (line[1] + '.')[:-1]
        #         paired = [False for i in range(len(line[1]))]
        #         # print(id(raw_sentence), id(line[1]), id(sentence))
        #         # pair long words with higher priority
        #         for token in sorted(dict, key=len, reverse=True):
        #             index = raw_sentence.find(token)
        #             # print(index, line[1])
        #             if index != -1 and not paired[index]:
        #                 line[1] = line[1].replace(token, " " + token + " ")
        #                 # mark as paired
        #                 for i in range(len(token)):
        #                     paired[index + i] = True
        #         # add sos
        #         tokens = [dict['<sos>']]
        #         for token in line[1].split():
        #             if token in dict:
        #                 tokens.append(dict[token])
        #             else:
        #                 unknown.add(token)
        #         # add eos
        #         tokens.append(dict['<eos>'])
        #         corpus[line[0]] = tokens
        # except Exception as e:
        #     raise
        # # add padding
        # length = [len(tokens) for key, tokens in corpus.items()]
        # max_length = max(length)
        # # print(max(length))
        # for key, tokens in corpus.items():
        #     if len(tokens) < max_length:
        #         tokens.extend([dict['<pad>']] * (max_length - len(tokens)))

    def test(self):
        self.model.eval()
        with torch.no_grad():
            target = np.arange(12).reshape(1, 12)
            target = torch.LongTensor(target)
            imgs = self.images.to(self.device)
            target = target.to(self.device)
            # print(imgs.shape)
            # forward(no teacher forcing)
            outputs = self.model(imgs, target, 0)

            # target: (batch_size, trg len)
            # outputs: (trg_len, batch_size, output_dim)
            # skip sos
            output_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, output_dim)
            target = target.permute(1, 0)[1:].reshape(-1)

            # compute the accuracy
            prediction = torch.max(outputs, 1)[1]
            target_label, prediction_label = target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy()
            score = accuracy_score(target_label, prediction_label)

            target_label = self.remove_endzeros_list(target_label)
            prediction_label = self.remove_endzeros_list(prediction_label)
            self.out_word = ' '.join([self.wold_dict[x] for x in prediction_label])
            # print('*************************state*******************')
            # print('target:', target_label)
            # print('prediction:', prediction_label)
            # print("score:", score)
            # print('target_wold:', ' '.join([self.wold_dict[x] for x in target_label]))
            # print('prediction_wold:', self.out_word)
            # print('*************************end*********************')



if __name__ == '__main__':
    # test()
    test = SignLanguagePredict()
    # start=time.time()
    test.read_images(image_path)
    test.test()
    # spent=time.time()-start
    # print(spent)