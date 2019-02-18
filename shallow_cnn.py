import time
import numpy as np
import keras.backend as K
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import Adam, Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical

# Parameters
NUM_EPOCHS = 30
BATCH_SIZE = 10
LEARNING_RATE = 0.005
ALPHA = 0.01
FILTERS = 50
CH = 2 # Channel (EEG is 2)
def prepare_data():
    """
    Helper function to prepare and split data
    """
    # Load EEG signals
    X = np.hstack((np.array(loadmat('psg_db/slp01am.mat')['val'])[CH][:997500],
                   np.array(loadmat('psg_db/slp01bm.mat')['val'])[CH][:997500],
                   np.array(loadmat('psg_db/slp02am.mat')['val'])[CH][:37500],
                   np.array(loadmat('psg_db/slp02am.mat')['val'])[CH][52500:225000],
                   np.array(loadmat('psg_db/slp02am.mat')['val'])[CH][232500:675000],
                   np.array(loadmat('psg_db/slp02am.mat')['val'])[CH][690000:705000],
                   np.array(loadmat('psg_db/slp02am.mat')['val'])[CH][712500:840000],
                   np.array(loadmat('psg_db/slp02am.mat')['val'])[CH][847500:885000],
                   np.array(loadmat('psg_db/slp02am.mat')['val'])[CH][892500:997500],
                   np.array(loadmat('psg_db/slp02bm.mat')['val'])[CH][:697500],
                   np.array(loadmat('psg_db/slp02bm.mat')['val'])[CH][705000:997500],
                   np.array(loadmat('psg_db/slp03m.mat')['val'])[CH][:997500],
                   np.array(loadmat('psg_db/slp04m.mat')['val'])[CH][:997500],
                   np.array(loadmat('psg_db/slp14m.mat')['val'])[CH][45000:540000],
                   np.array(loadmat('psg_db/slp14m.mat')['val'])[CH][547500:997500],
                   np.array(loadmat('psg_db/slp16m.mat')['val'])[CH][195000:997500],
                   np.array(loadmat('psg_db/slp32m.mat')['val'])[CH][:997500],
                   np.array(loadmat('psg_db/slp37m.mat')['val'])[CH][15000:997500],
                   np.array(loadmat('psg_db/slp41m.mat')['val'])[CH][:997500],
                   np.array(loadmat('psg_db/slp45m.mat')['val'])[CH][:997500],
                   np.array(loadmat('psg_db/slp48m.mat')['val'])[CH][:465000],
                   np.array(loadmat('psg_db/slp48m.mat')['val'])[CH][472500:997500],
                   np.array(loadmat('psg_db/slp59m.mat')['val'])[CH][165000:997500],
                   np.array(loadmat('psg_db/slp60m.mat')['val'])[CH][:997500],
                   np.array(loadmat('psg_db/slp61m.mat')['val'])[CH][150000:997500],
                   np.array(loadmat('psg_db/slp66m.mat')['val'])[CH][:997500],
                   np.array(loadmat('psg_db/slp67xm.mat')['val'])[CH][:997500]))
    M = 2307
    X = np.reshape(X, (M, 7500))

    # Load annotations as labels
    annotations = open('psg_db/annotations.txt').readlines()
    Y = []

    for i in range(M):
        stage = annotations[i].split('\t')[1][0]
        if stage == 'W':
            stage = 0;
        elif stage == 'R':
            stage = 5
        else:
            stage = int(stage)
        Y.append(stage)

    Y = to_categorical(Y)

    # Shuffle data
    data = np.hstack((X,Y))
    np.random.shuffle(data)
    X = data[:, :7500]
    Y = data[:, 7500:]

    # Split data
    N = int(0.7*M)
    x_train = np.expand_dims(X[:N], axis=2)
    y_train = Y[:N]
    x_test = np.expand_dims(X[N:], axis=2)
    y_test = Y[N:]

    return x_train, y_train, x_test, y_test

def network(input_shape):
    """
    Network Architecture
    --------------------
    x: Main network layers
    c: Concatenate nodes
    s: Skip network layers
    ---------------------
    """
    # Input Normalization
    input = Input(shape=input_shape)
    c = BatchNormalization(input_shape=input_shape, center=False, scale=True)(input)
    # Conv Branch 1
    x = Conv1D(filters=FILTERS, kernel_size=125, activation= 'tanh')(c)
    x = MaxPool1D(pool_size=2)(x)
    # Conv Branch 2
    y = Conv1D(filters=FILTERS, kernel_size=15, activation='tanh')(c)
    y = MaxPool1D(pool_size=2)(y)
    # Conv Branch 3
    z = Conv1D(filters=FILTERS, kernel_size=5, activation='tanh')(c)
    z = MaxPool1D(pool_size=2)(z)
    # Conv Skip
    #s = LeakyReLU(alpha=ALPHA)(c)
    #s = Dense(FILTERS)(s)
    #s = Dropout(0.2)(s)
    s = c
    # Concatentate
    c = Concatenate(axis=1)([x,y,z])
    c = MaxPool1D(pool_size=3)(c)
    c = Lambda(lambda t: K.repeat_elements(t,2,axis=1))(c)
    c = ZeroPadding1D(padding=(24,24))(c)
    c = Dropout(0.2)(c)
    c = Add()([c,s])
    # Output
    x = GlobalAvgPool1D()(x)
    output = Dense(6, activation='softmax')(x)

    model = Model(inputs=[input], outputs=[output])

    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train(x_train, y_train, x_test, y_test):
    """
    Helper function to train network
    """
    save_point = ModelCheckpoint(filepath='saved_models/psg_shallowmodel.h5', monitor='val_loss', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    callbacks = [save_point]

    clf = network(input_shape=(7500,1))
    clf.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks)

if __name__ == "__main__":
    start = time.time()

    x,y,xt,yt = prepare_data()

    train(x,y,xt,yt)

    T = int(time.time()-start)
    print("\nFinished training in " + str(int(T/60)) + " minutes and " + str(T%60) + " seconds.")

    try:
        classifier = load_model("saved_models/psg_shallowmodel.h5")

        print("\nFinal evaluation over training samples")
        scores_train = classifier.evaluate(x, y, BATCH_SIZE)
        print("Loss: ", scores_train[0])
        print("Accuracy: ", scores_train[1])

        print("\nFinal evaluation over test samples")
        scores_test = classifier.evaluate(xt, yt, BATCH_SIZE)
        print("Loss: ", scores_test[0])
        print("Accuracy: ", scores_test[1])

        print("\nFinal evaluation over all samples")
        X = np.concatenate((x, xt), axis=0)
        Y = np.concatenate((y, yt), axis=0)
        scores_full = classifier.evaluate(X, Y, BATCH_SIZE)
        print("Loss: ", scores_full[0])
        print("Accuracy: ", scores_full[1])
    except Exception as e:
        print("Couldn't load model for final evaluation")
        print(e)
    print("finished")
