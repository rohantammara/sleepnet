import time
import numpy as np
from keras.models import load_model, Model
from sklearn.metrics import cohen_kappa_score
from main import prepare_data, BATCH_SIZE

TRIALS = 20

print("\nStarting test (" + str(TRIALS) + " trials)\n")

start = time.time()

avg_loss = 0
avg_acc = 0
best_loss = 0
best_acc = 0

x,y,xt,yt = prepare_data()

classifier = load_model('saved_models/psg_model_2.h5')

for _ in range(TRIALS):
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

avg_loss = avg_loss/TRIALS
avg_acc = avg_acc/TRIALS

print("\nBest Loss: ", best_loss)
print("Best Accuracy: ", best_acc)
print("\nMean Loss: ", avg_loss)
print("Mean Accuracy: ", avg_acc)
print("\nCohen's Kappa Score: ", kappa)
print("\nfinished in " + str(time.time()-start) + " seconds")
