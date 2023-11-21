from sklearn import datasets,svm
from sklearn.model_selection import train_test_split

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X,y
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

def split_data(X,y,test_size,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5)
    return X_train,X_test,y_train,y_test

def train_model(X,y,model_params,model_type = 'svm'):
    if model_type == "svm":
        clf = svm.SVC
    model = clf(**model_params)
    # Learn the digits on the train subset
    model.fit(X,y)
    return model

def train_test_dev_split():
    pass
def predict_and_eval():
    pass