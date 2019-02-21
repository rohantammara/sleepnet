import time
import numpy as np
from keras.models import load_model, Model
from sklearn.metrics import cohen_kappa_score
from deep_cnn_l2 import prepare_data, BATCH_SIZE

TRIALS = 15

print("\nStarting test [" + str(TRIALS) + " trials]\n(This might take a while ...)\n")

start = time.time()

avg_loss = 0
avg_acc = 0
best_loss = 0
best_acc = 0
best_kappa = 0
worst_kappa = 1

classifier = load_model('saved_models/psg_L2deepmodel.h5')

for _ in range(TRIALS):
    x,y,xt,yt = prepare_data()

    scores = classifier.evaluate(xt, yt, BATCH_SIZE, verbose=0)

    if scores[0] > best_loss:
        best_loss = scores[0]
    if scores[1] > best_acc:
        best_acc = scores[1]
    avg_loss += scores[0]
    avg_acc += scores[1]

    yp = classifier.predict(xt, verbose=0)
    y_test = np.argmax(yt, axis=1)
    y_pred = np.argmax(yp, axis=1)

    kappa = cohen_kappa_score(y_test, y_pred)
    if kappa > best_kappa:
        best_kappa = kappa
    if kappa < worst_kappa:
        worst_kappa = kappa

avg_loss = avg_loss/TRIALS
avg_acc = avg_acc/TRIALS

print("\nBest Loss: ", best_loss)
print("Best Accuracy: ", best_acc)
print("\nMean Loss: ", avg_loss)
print("Mean Accuracy: ", avg_acc)
print("\nCohen's Kappa Score")
print("Best k: ", best_kappa)
print("Worst k: ", worst_kappa)

T = int(time.time()-start)
print("\nFinished validating in " + str(int(T/60)) + " minutes and " + str(T%60) + " seconds.")
