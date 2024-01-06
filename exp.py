"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports 
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics

from utils import preprocess_data,split_data,train_model,read_digits,train_test_dev_split,predict_and_eval

gamma_ranges = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
C_ranges = [0.1, 1, 10, 100, 1000]


#  3 . Splitting the data
# Split data into 50% train and 50% test subsets
X,y = read_digits()
X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X,y,test_size=0.3,dev_size=0.2)

# 4. Preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)
# flatten the images

# Hyperparameter tuning 

best_acc_so_far = -1
best_model = None
for cur_gamma in gamma_ranges:
    for cur_c in C_ranges:
        #print("Running for gamma={} c = {}".format(cur_gamma,cur_c))
        cur_model = train_model(X_train,y_train,{'gamma': cur_gamma, 'C': cur_c},model_type='svm')
        cur_accuracy = predict_and_eval(cur_model,X_dev,y_dev)
        if cur_accuracy>best_acc_so_far:
            print("New best accuracy=",cur_accuracy)
            best_acc_so_far = cur_accuracy
            Optimal_C = cur_c
            Optimal_gamma = cur_gamma
            best_model = cur_model
print("Optimal gamma = {} Optimal_C = {}".format(Optimal_gamma,Optimal_C))       
#  5. Training the data

model = train_model(X_train,y_train,{'gamma': Optimal_gamma, 'C': Optimal_C},model_type='svm')



# 6. Getting model prediction on test data
# Predict the value of the digit on the test subset
test_accuracy = predict_and_eval(model,X_test,y_test)
print("Test accuracy = ",test_accuracy)

###############################################################################

# 7 . Qualititative sanity check of the predictions
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

