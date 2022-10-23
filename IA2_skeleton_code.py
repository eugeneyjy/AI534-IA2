# AI534
# IA2 skeleton code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(179)

# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    loaded_data = pd.read_csv(path)
    return loaded_data

# Implements dataset preprocessing. For this assignment, you just need to implement normalization 
# of the three numerical features.
def preprocess_data(data):
    # Your code here:
    train = data[0].copy()
    val = data[1].copy()
    # Columns to normalize
    norm_col = ['Age', 'Annual_Premium', 'Vintage']
    # Store normalize info on training data to preserve normalize scale
    norm_info = (train[norm_col].mean(), train[norm_col].std())
    # Normalize training and validation data on the same scale
    train[norm_col] = train[norm_col].apply(lambda x: (x - norm_info[0][x.name])/norm_info[1][x.name])
    val[norm_col] = val[norm_col].apply(lambda x: (x - norm_info[0][x.name])/norm_info[1][x.name])
    return (train, val)

def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def negative_loglikelihood(X, y, w, lambd):
    size = X.shape[0]
    reg = lambd*(np.square(w[1:]).sum())
    sig = sigmoid(X@w)
    lw = (-y.T@np.log(sig) - (1-y).T@np.log(1-sig))/size
    return (lw + reg)[0][0]

def calc_accuracy(X, y, w):
    size = X.shape[0]
    sig = sigmoid(X@w)
    pred = sig > 0.5
    acc = np.sum(pred == y)/size
    return acc

def calc_sparsity(weights):
    threshold = 10e-6
    return np.sum(weights <= threshold)

# Trains a logistic regression model with L2 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L2_train(train_data, val_data, lr, lambd, max_epoch):
    # Your code here:
    y_train = train_data.iloc[:, -1].to_numpy()[:, np.newaxis]
    X_train = train_data.iloc[:, :-1].to_numpy()
    y_val = val_data.iloc[:, -1].to_numpy()[:, np.newaxis]
    X_val = val_data.iloc[:, :-1].to_numpy()
    size = X_train.shape[0]
    dims = X_train.shape[1]
    weights = np.random.rand(dims, 1)
    losses = []
    # epsilon = 10e-5
    for epoch in range(max_epoch):
        gradients = (1/size)*((sigmoid(X_train@weights)-y_train).T@X_train).T
        penalty = lambd*weights[1:]
        grad_norm = np.linalg.norm(gradients) + np.linalg.norm(penalty)
        weights -= lr*gradients
        weights[1:] -= lr*penalty
        loss = negative_loglikelihood(X_train, y_train, weights, lambd)
        losses.append(loss)
        # Stop training if loss starts to go up
        if epoch > 0 and loss > losses[-2]:
            break
        print(f"lambda: {lambd}, epoch: {epoch}, loss: {loss}, gradient_norm: {grad_norm}")
    train_acc = calc_accuracy(X_train, y_train, weights)
    val_acc = calc_accuracy(X_val, y_val, weights)
    print(f"train_acc: {train_acc}, val_acc: {val_acc}")
    return weights, train_acc, val_acc

# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L1_train(train_data, val_data, lambd):
    # Your code here:

    return weights, train_acc, val_acc

# Generates and saves plots of the accuracy curves. Note that you can interpret accs as a matrix
# containing the accuracies of runs with different lambda values and then put multiple loss curves in a single plot.
def plot_losses(accs):
    # Your code here:
    # for lr in losses:
    plt.plot(accs[:, 0], accs[:, 1], label=f"Train Acc")
    plt.plot(accs[:, 0], accs[:, 2], label="Val Acc")
    plt.ylabel("Accuracy")
    plt.xlabel("Lambda")
    plt.legend()
    plt.show()
    plt.savefig("losses_graph.png")
    accs_df = pd.DataFrame(accs, columns=["lambda", "train", "val"])
    accs_df.to_csv("accuracies.csv", index=False)
    return

def plot_sparsity(all_weights, lambds, path):
    sparsities = []
    for i in range(len(lambds)):
        sparsities.append(calc_sparsity(all_weights[i]))
    sparsities_df = pd.DataFrame(sparsities, index=lambds, columns=["sparsity"])
    sparsities_df.index.name = "lambda"
    sparsities_df.to_csv(path)
    return

def save_weights(labels, all_weights, path):
    weights_df = pd.DataFrame(all_weights, columns=labels)
    weights_df.to_csv(path, index=False)
    return

def save_top_5_features(labels, all_weights, lambds, path):
    features = []
    for weights in all_weights:
        top_5_idx = np.argpartition(np.absolute(weights[1:]), -5, axis=0)[-5:]
        features.append(labels[top_5_idx+1])
    features_df = pd.DataFrame(features, index=lambds, columns=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"])
    features_df.index.name = "lambda"
    features_df.to_csv(path)
    return

# Invoke the above functions to implement the required functionality for each part of the assignment.
# Part 0  : Data preprocessing.
# Your code here:
train_data = load_data('IA2-train.csv')
val_data = load_data('IA2-dev.csv')
norm_train, norm_val = preprocess_data((train_data, val_data))

# Part 1 . Implement logistic regression with L2 regularization and experiment with different lambdas
# Your code here:
lambds = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2]
lrs = [2.5, 2.5, 2, 2, 1, 0.01, 0.01, 0.01]
accs = []
all_weights = []

for i in range(len(lambds)):
    max_epoch = 5000
    weights, train_acc, val_acc = LR_L2_train(norm_train, norm_val, lrs[i], lambds[i], max_epoch)
    accs.append([lambds[i], train_acc, val_acc])
    all_weights.append(weights.flatten())

accs = np.array(accs)
plot_losses(accs)
save_weights(norm_train.columns[:-1], all_weights, "weights_p1.csv")
save_top_5_features(norm_train.columns[:-1], all_weights, lambds, "top_features_p1.csv")
plot_sparsity(all_weights, lambds, "sparcities_p1.csv")

# Part 2  Training and experimenting with IA2-train-noisy data.
# Your code here:


# Part 3  Implement logistic regression with L1 regularization and experiment with different lambdas
# Your code here:



