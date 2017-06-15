import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import numpy as np
import sys
import matplotlib.pyplot as plt
csv_lines = open('2014-0723-1342.txt').readlines()[3:]
print ('day count = ' + str(len(csv_lines)))
print (csv_lines[0])
former_all = []
latter_all = []

def plot_result(predicted_data, true_data):
    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label = 'true data')
    plt.plot(predicted_data, label = 'prediction')
    plt.legend()
    plt.show()

def plot_train(true_data):
    fig = plt.figure(facecolor = 'white')
    plt.plot(true_data, label = 'training_set')
    plt.legend()
    plt.show()

for line in csv_lines:
    item = line.split()
    F = [float(item[11]), float(item[13])-14, float(item[14])]
    if len(former_all) == 0:
        F.append(0.00001)
        F.append(0.00001)
        F.append(0.00001)
    else:
        last = former_all[-1]
        F.append((F[0]-last[0]))
        F.append((F[1]-last[1]))
        F.append((F[2]-last[2]))

    former_all.append(F)
    L = F[0]  ###3->0
    latter_all.append(L)

print len(former_all)
print len(latter_all)

former = former_all[160000:200000]
latter = latter_all[160000:200000]
former_test = former_all[200000:205000]
latter_test = latter_all[200000:205000]

maxlen = 1
step = 1
sequences = []
next_days = []

for i in range(0, len(former) - maxlen, step):
    sequences.append(former[i: i+maxlen])
    next_days.append(latter[i+maxlen])
print('number of sequences:', len(sequences))


sequences_test = []
next_days_test = []
#regression for the stock price up/down %chg
for i in range(0, len(former_test) - maxlen, step):
    sequences_test.append(former_test[i: i+maxlen])
    next_days_test.append(latter_test[i+maxlen])
print('number of test sequences:', len(sequences_test))

print next_days_test

print('getting training vector...')
X = np.zeros((len(sequences), maxlen, 6), dtype=np.float32)
y = np.zeros((len(sequences), 1), dtype=np.float32)
for i, sequence in enumerate(sequences):
    for t, day in enumerate(sequence):
        for g in xrange(0,6):
            X[i, t, g] = day[g]
    y[i, 0] = next_days[i]


model = Sequential()
model.add(LSTM(1024, return_sequences = True, input_shape=(maxlen, 6)))
model.add(LSTM(1024, return_sequences = False))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mse', optimizer='rmsprop')

for iteration in range(1, 2):
    print('------Iteration------', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)
    predicted = []
    truedata = []
    start = datetime.datetime.now()
    for seq,tar in zip(sequences_test[:5000], next_days_test[:5000]):
        x = np.zeros((1, maxlen, 6))
        for t, day in enumerate(seq):
            for g in xrange(0,6):
                x[0, t, g] = day[g]
        
        preds = model.predict(x, verbose=0)[0]
        predicted.append(preds[0])
        truedata.append(tar)
#        print(preds[0], tar)
    end = datetime.datetime.now()
    print (end-start)
    plot_result(predicted,truedata)
        
