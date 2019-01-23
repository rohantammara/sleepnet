import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from keras.models import Model, load_model
from keras.layers import Input, LeakyReLU, Dense, Dropout, Conv1D, MaxPool1D, \
                         AvgPool1D, GlobalAvgPool1D, Flatten, BatchNormalization, \
                         Concatenate
from keras.optimizers import Adam, SGD
from keras.constraints import MinMaxNorm
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical

# Parameters
NUM_EPOCHS = 50
BATCH_SIZE = 10
LEARNING_RATE = 0.005
ALPHA = 0.05

def prepare_data(blocks=2):
    """
    Helper function to prepare and split data
    """
    # Load EEG signals
    if  blocks == 2:
        X = np.hstack((np.array(loadmat('psg_db/slp01am.mat')['val'])[2][:997500],
                       np.array(loadmat('psg_db/slp01bm.mat')['val'])[2][:997500]))
        M = 266
    elif blocks == 1:
        X = np.array(loadmat('psg_db/slp01am.mat')['val'])[2][:997500]
        M = 133
    X = np.reshape(X, (M, 7500))

    # Load annotations as labels
    annotations = open('psg_db/annotations.txt').readlines()
    Y = []

    for i in range(M):
        stage = annotations[i].split('\t')[1][0]
        if stage == 'W':
            stage = 0;
        elif stage == 'R':
            stage = 6
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
    """
    # Auxiliaries
    beta_constraint = MinMaxNorm(min_value=0.0, max_value=0.0)
    gamma_constraint = MinMaxNorm(min_value=1.0, max_value=1.0)
    # Normalization
    input = Input(shape=input_shape)
    normed = BatchNormalization(input_shape=input_shape, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint)(input)
    # Convolution
    a = Conv1D(filters=32, kernel_size=20)(normed)
    a = LeakyReLU(alpha=ALPHA)(a)
    a = Conv1D(filters=32, kernel_size=20)(a)
    a = LeakyReLU(alpha=ALPHA)(a)
    a = MaxPool1D(pool_size=2)(a)
    a = Dropout(0.2)(a)
    # Convolution
    b = Conv1D(filters=32, kernel_size=10)(normed)
    b = LeakyReLU(alpha=ALPHA)(b)
    b = Conv1D(filters=32, kernel_size=10)(b)
    b = LeakyReLU(alpha=ALPHA)(b)
    b = MaxPool1D(pool_size=2)(b)
    b = Dropout(0.2)(b)
    # Convolution
    c = Conv1D(filters=32, kernel_size=5)(normed)
    c = LeakyReLU(alpha=ALPHA)(c)
    c = Conv1D(filters=32, kernel_size=5)(c)
    c = LeakyReLU(alpha=ALPHA)(c)
    c = MaxPool1D(pool_size=2)(c)
    c = Dropout(0.2)(c)
    # Merge\Pool
    x = Concatenate(axis=1)([a,b,c])
    x = GlobalAvgPool1D()(x)
    # Output
    output = Dense(5, activation='softmax')(x)

    model = Model(inputs=[input], outputs=[output])

    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train(x_train, y_train, x_test, y_test):
    """
    Helper function to train network
    """
    save_point = ModelCheckpoint(filepath='saved_models/psg_model.hdf5', monitor='val_loss', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    callbacks = [save_point]

    clf = network(input_shape=(7500,1))
    clf.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks)

if __name__ == "__main__":

    x,y,xt,yt = prepare_data(blocks=2)

    train(x,y,xt,yt)

    clf = load_model("saved_models/psg_model.hdf5")

    print("\nFinal evaluation over training samples")
    scores_70 = clf.evaluate(x, y, BATCH_SIZE)
    print("Loss: ", scores_70[0])
    print("Accuracy: ", scores_70[1])

    print("\nFinal evaluation over test samples")
    scores_30 = clf.evaluate(xt, yt, BATCH_SIZE)
    print("Loss: ", scores_30[0])
    print("Accuracy: ", scores_30[1])

    print("\nFinal evaluation over all samples")
    X = np.concatenate((x, xt), axis=0)
    Y = np.concatenate((y, yt), axis=0)
    scores_full = clf.evaluate(X, Y, BATCH_SIZE)
    print("Loss: ", scores_full[0])
    print("Accuracy: ", scores_full[1])
