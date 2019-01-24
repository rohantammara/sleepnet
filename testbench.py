from time import time
from keras.models import load_model
from main import prepare_data, BATCH_SIZE

TRIALS = 10

print("\nStarting test (" + str(TRIALS) + " trials)\n")

start = time()
loss = 0
acc = 0

x,y,xt,yt = prepare_data()

classifier = load_model('saved_models/psg_model_2.h5')

for _ in range(TRIALS):
    scores = classifier.evaluate(xt,yt,BATCH_SIZE,verbose=0)
    loss += scores[0]
    acc += scores[1]

loss = loss/TRIALS
acc = acc/TRIALS

print("\nLoss: ", loss)
print("Accuracy: ", acc)
print("\nfinished in " + str(time()-start) + " seconds")
