import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import os

cuda_enabled = torch.cuda.is_available()

class swbd_data(Dataset):


    def __init__(self, dataset, label, padlen=660, featlen=40, encoder=None):
        self.pad_len = padlen
        self.feat_len = featlen
        self.dataset = dataset   # train, test, dev

        # Persist the encoder for possible future use.
        self.__label_encoder__ = encoder
        # Persist a file list -- may need it for printing & reporting
        self.file_descriptions = None

        self.instance_list = []
        self.instance_label = []
        self.__longest_vector__ = 0
        self.__load_features_and_labels__(self.__get_label_column__(label))


    def get_files(self):
        return self.file_descriptions


    def get_encoder(self):
        return self.__label_encoder__


    def __get_label_column__(self, label):
        if label == 'speaker':
            return 2
        if label == 'gender':
            return 5
        if label == 'region':
            return 6
        if label == 'generation':
            return 3
        if label == 'decade':
            return 4
        if label == 'year':
            return 7
        if label == 'education':
            return 10


    def __get_pad_length__(self):
        # TODO: Replace from data
        return self.pad_len
        #return 2939


    def __get_swbd_material__(self):
        import csv

        label_file = '/home/dgbrizan/project/CALS_tools/file_listing.csv'
        descriptor_list = []
        all_files = []

        with open(label_file) as f:
            reader = csv.reader(f)
            all_files = list(reader)
        for line in all_files:
            if 'Switchboard' in line[0] and line[1].strip() == self.dataset:
                if os.path.isfile(line[0]):
                    descriptor_list.append(line)
        return descriptor_list


    def __get_mfcc_matrix__(self, file_name):
        import librosa

        #wav, sr = librosa.load(file_name, mono=True)
        wav, sr = librosa.load(file_name, sr=None, mono=True)
        mfcc = librosa.feature.mfcc(wav, sr, n_mfcc=self.feat_len)
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
        return mel_spectrum
 

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

        self.file_descriptions = self.__get_swbd_material__()
        file_label = []

        for description in self.file_descriptions:
            unpadded = self.__get_mfcc_matrix__(description[0])
            #unpadded = self.__get_melfilter_bank_matrix__(description[0])
            if len(unpadded[0]) > self.__longest_vector__:
                self.__longest_vector__ = len(unpadded[0])
            file_label.append(description[label_column])
            self.instance_list.append(unpadded)

        # Change "X" to padded version
        self.instance_list = self.__pad_data__(self.instance_list)

        # Encode "Y"
        if self.__label_encoder__ == None:
            self.__label_encoder__ = sklearn.preprocessing.LabelEncoder()
        self.__label_encoder__.fit(file_label)
        self.instance_label = self.__label_encoder__.transform(file_label).flatten()


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
        #self.fc = nn.Dropout(p=0.75, inplace=False)
        self.fc = nn.Dropout(p=0.5, inplace=False)
        self.linear = nn.Linear(self.hidden_size*2, self.num_classes)

        if cuda_enabled:
            self.lstm = self.lstm.cuda()
            self.fc = self.fc.cuda()
            self.linear = self.linear.cuda()

    def forward(self, x):
        # Set initial states
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
    import datetime
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Keep timing -- for future reporting
    timing = dict()

    # Data description
    input_size = 660  # 2939, 20 # TODO: Determine this from the data
    sequence_length = 40  # TODO: Determine this from the data
    # Data
    label = 'gender'  # gender, region, speaker

    timing['start'] = datetime.datetime.now()

    train = swbd_data('train', label, padlen=input_size, featlen=sequence_length)
    print('Switchboard train data items: {}'.format(train.__len__()))

    dev = swbd_data('dev', label, encoder=train.get_encoder())
    print('Switchboard dev data items: {}'.format(dev.__len__()))

    test = swbd_data('test', label, encoder=train.get_encoder())
    print('Switchboard test data items: {}'.format(test.__len__()))

    timing['features'] = datetime.datetime.now() - timing['start']
    print('Extracted features from speech files.')

    # Some hyperparams, etc. for the network
    batch_size = 64
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if not cuda_enabled:
        kwargs['pin_memory'] = False
        batch_size = 32

    print('Starting loader')
    timing['training'] = datetime.datetime.now()
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, **kwargs)

    # The net... and training it
    hidden_size = 128
    num_layers = 2
    num_classes = 2  # TODO: Determine this from the data
    learning_rate = 0.0001
    num_epochs = 20000

    # The network
    rnn = BiRNN(input_size, hidden_size, num_layers, num_classes)
    rnn.is_training = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    epoch_loss = 5000.
    # Train it
    for epoch in range(num_epochs):
        loss_total = 0.
        iteration_count = 0.
        for i, (mfcc, labels) in enumerate(train_loader):
            iteration_count += 1.
            mfcc = Variable(mfcc.view(-1, sequence_length, input_size))
            labels = Variable(labels)
            if cuda_enabled:
                mfcc = mfcc.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(mfcc)

            loss = criterion(outputs, labels)
            loss_total += loss.data[0]
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train) // batch_size, loss.data[0]))
        current_epoch_loss = loss_total / iteration_count
        # Optimise training epochs: only continue training while the loss drops
        if current_epoch_loss >= epoch_loss:
            break
        epoch_loss = current_epoch_loss

    timing['training'] = datetime.datetime.now() - timing['training']

    # Test the Model
    rnn.is_training = False
    timing['testing'] = datetime.datetime.now()
    print('Testing -----------------------------------------------')
    correct = 0.0
    total = 0.0
    predicted_list = []
    label_list = []
    for mfcc, labels in test_loader:#test_loader
        mfcc = Variable(mfcc.view(-1, sequence_length, input_size))
        if cuda_enabled:
            mfcc = mfcc.cuda()

        outputs = rnn(mfcc)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        for p, l in zip(predicted, labels):
            predicted_list.append(p)
            label_list.append(l)
            if p == l:
                correct += 1.0
    timing['testing'] = datetime.datetime.now() - timing['testing']

    print('')
    print('Writing predictions to predictions.csv\n')
    of = open('predictions.csv', 'w')
    of.write('Predicted,Actual,File,Material,Speaker,Gender,Region\n')
    for f, a, p in zip(test.get_files(), label_list, predicted_list):
        of.write(str(p) + ',' + str(a) + ',' + ','.join(f) + '\n')
    of.close()

    print('Timing (feature extraction, training, timing)')
    print('=============================================')
    print(timing['features'])
    print(timing['training'])
    print(timing['testing'])
    print('')
    print('=============================================')
    print('')
    print('Confusion Matrix')
    print('================')
    print(train.get_encoder().classes_)
    print(confusion_matrix(label_list, predicted_list))
    print('=============================================')
    print('Accuracy = %0.4f' % (accuracy_score(label_list, predicted_list)))
    print('=============================================')
    

    # Save the Model
    # torch.save(rnn.state_dict(), 'swbd.pkl')

if __name__ == '__main__':
    main_biRNN()



