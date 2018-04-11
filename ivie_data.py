import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import os

cuda_enabled = torch.cuda.is_available()
#cuda_enabled = False

class ivie_data(Dataset):


    def __init__(self, dataset, label, padlen=2939, featlen=40):
        self.pad_len = padlen
        self.feat_len = featlen
        self.dataset = dataset   # train, test, dev
        self.instance_list = []
        self.instance_label = []
        self.__longest_vector__ = 0
        self.__load_features_and_labels__(self.__get_label_column__(label))


    def __get_label_column__(self, label):
        if label == 'speaker':
            return 2
        if label == 'gender':
            return 3
        if label == 'region':
            return 4


    def __get_pad_length__(self):
        # TODO: Replace from data
        return self.pad_len
        #return 2939


    def __get_ivie_material__(self):
        import csv

        label_file = 'listing.csv'
        descriptor_list = []
        all_files = []

        with open(label_file) as f:
            reader = csv.reader(f)
            all_files = list(reader)
        for line in all_files:
            if 'IViE' in line[0] and line[1].strip() == self.dataset:
                if os.path.isfile(line[0]):
                    descriptor_list.append(line)
        return descriptor_list


    def __get_mfcc_matrix__(self, file_name):
        import librosa

        #wav, sr = librosa.load(file_name, mono=True)
        wav, sr = librosa.load(file_name, sr=None, mono=True)
        mfcc = librosa.feature.mfcc(wav, sr, n_mfcc=self.feat_len)
        #print("mfcc num: {}, feat dim: {}".format(len(mfcc[0]), len(mfcc)))
        return mfcc

    def __get_melfilter_bank_matrix__(self, file_name):
        import librosa
        import numpy
        import scipy
        eps = numpy.spacing(1)
        wav, sr = librosa.load(file_name, sr=None, mono=True)
        n_fft = 512
        win_len = 512
        hop_len = 160
        n_mels = 40
        htk = False
        upper_f = 6855.4976
        lower_f = upper_f / 4
        window = scipy.signal.hamming(n_fft, sym=True)

        magnitude_spectrogram = numpy.abs(
            librosa.stft(wav + eps, n_fft=n_fft, win_length=win_len,
                         hop_length=hop_len, window=window)) ** 2
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                        fmin=lower_f, fmax=upper_f, htk=htk)
        mel_spectrum = numpy.dot(mel_basis, magnitude_spectrogram)
        #print("len: {}, dim, {}".format(len(mel_spectrum[0]), len(mel_spectrum)))
        return mel_spectrum
        #mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum))

    def __pad_data__(self, series):
        import numpy as np

        padded = []
        for i in range(len(series)):
            row = np.zeros((self.feat_len, self.__get_pad_length__()), dtype=np.float32)
            for j in range(self.feat_len):
                for k in range(len(series[i][j])):
                    row[j][k] = series[i][j][k]
            padded.append(row)
        return padded


    def __load_features_and_labels__(self, label_column):
        import sklearn.preprocessing

        file_descriptions = self.__get_ivie_material__()
        file_label = []

        for description in file_descriptions:
            unpadded = self.__get_mfcc_matrix__(description[0])
            #unpadded = self.__get_melfilter_bank_matrix__(description[0])
            if len(unpadded[0]) > self.__longest_vector__:
                self.__longest_vector__ = len(unpadded[0])
            file_label.append(description[label_column])
            self.instance_list.append(unpadded)

        # Change "X" to padded version
        self.instance_list = self.__pad_data__(self.instance_list)

        # Change "Y" to one-hot encoding:
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(file_label)
        self.instance_label = label_encoder.transform(file_label).flatten()
        '''
        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        label_binarizer.fit(file_label)
        self.instance_label = label_binarizer.transform(file_label).flatten() 
        '''


    def __getitem__(self, index):
        return self.instance_list[index], self.instance_label[index]


    def __len__(self):
        return len(self.instance_list)

# Copied from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
class BiRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.is_training = False
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Dropout(p=0.5, inplace=False)
        self.linear = nn.Linear(self.hidden_size*2, self.num_classes)

        #self.fc = nn.Dropout(p=0.75, inplace=False)
        if cuda_enabled:
            self.lstm = self.lstm.cuda()
            self.fc = self.fc.cuda()
            self.linear = self.linear.cuda()

    def forward(self, x):
        # Set initial states
        #h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()) # 2 for bidirection
        #c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda())
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        if cuda_enabled:
            h0 = h0.cuda()  # 2 for bidirection
            c0 = c0.cuda()

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode hidden state of last time step
        if self.is_training:
            out = self.fc(out[:, -1, :])
        else:
            out = out[:, -1, :]

        out = F.log_softmax(self.linear(out), dim=1)
        return out

def main_biRNN():
    # Data description
    input_size = 2939  # 2939, 20 # TODO: Determine this from the data
    sequence_length = 40  # TODO: Determine this from the data
    # Data
    label = 'gender'  # gender, region, speaker

    train = ivie_data('train', label, padlen=input_size, featlen=sequence_length)
    print('IViE train data items: {}'.format(train.__len__()))
    dev = ivie_data('dev', label)
    print('IViE dev data items: {}'.format(dev.__len__()))
    test = ivie_data('test', label)
    print('IViE test data items: {}'.format(test.__len__()))

    # Some hyperparams, etc. for the network
    batch_size = 64
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if not cuda_enabled:
        kwargs['pin_memory'] = False
        batch_size = 32

    print('Starting loader')
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, **kwargs)

    # The net... and training it
    hidden_size = 128
    num_layers = 2
    num_classes = 9  # TODO: Determine this from the data
    learning_rate = 0.0001
    # num_epochs = 100  # For debugging; change to below for testing
    num_epochs = 3000

    # The network
    rnn = BiRNN(input_size, hidden_size, num_layers, num_classes)
    rnn.is_training = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Train it
    for epoch in range(num_epochs):
        for i, (mfcc, labels) in enumerate(train_loader):
            # mfcc = Variable(mfcc.view(-1, sequence_length, input_size).cuda())
            # labels = Variable(labels.cuda())
            mfcc = Variable(mfcc.view(-1, sequence_length, input_size))
            labels = Variable(labels)
            if cuda_enabled:
                mfcc = mfcc.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(mfcc)
            #outputs = outputs[:, -2:]
            # print("outputs: {}".format(outputs))
            # print("labels: {}".format(labels))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train) // batch_size, loss.data[0]))

    # Test the Model
    rnn.is_training = False
    print('Testing -----------------------------------------------')
    correct = 0.0
    total = 0.0
    for mfcc, labels in test_loader:#test_loader
        # mfcc = Variable(mfcc.view(-1, sequence_length, input_size).cuda())
        # outputs = rnn(mfcc).cuda()
        mfcc = Variable(mfcc.view(-1, sequence_length, input_size))
        #outputs = outputs[:, -2:]
        if cuda_enabled:
            mfcc = mfcc.cuda()

        outputs = rnn(mfcc)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # correct += (predicted == labels).sum()
        for p, l in zip(predicted, labels):
            if p == l:
                correct += 1.0

    print('total: {}, correct: {}'.format(total, correct))
    print('Test Accuracy of the model on the test speech: %2f %%' % (100.0 * correct / total))

    # Save the Model
    # torch.save(rnn.state_dict(), 'lstm.pkl')

if __name__ == '__main__':
    main_biRNN()



