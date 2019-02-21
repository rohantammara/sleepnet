import numpy as np
from keras.models import load_model, Model
from deep_cnn_l1 import prepare_data

# Load models
clf_1 = load_model('saved_models/psg_L1deepmodel_1.h5')
clf_2 = load_model('saved_models/psg_L2deepmodel.h5')

# Prepare data
x,y,xt,yt = prepare_data()

# Classify between Wake/NREM/REM
y1_pred = clf_1.predict(xt)
y1_pred = np.argmax(y1_pred, axis=1)

# Collect data points which are NREM
xt_2 = []
for i in range(len(y1_pred)):
    if y1_pred[i] == 1:
        xt_2.append(xt[i])
xt_2 = np.array(xt_2)

# Classify NREM points between stage 1/2/3/4
y2_pred = clf_2.predict(xt_2)
y2_pred = np.argmax(y2_pred, axis=1)

# Measure accuracy
print(y2_pred)
