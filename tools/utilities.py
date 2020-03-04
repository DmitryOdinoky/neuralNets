import numpy as np
import matplotlib.pyplot as plt  # import matplotlib for plotting and visualization
import matplotlib


def compute_cost(Y, Y_hat):
    """
    This function computes and returns the Cost and its derivative.
    The is function uses the Squared Error Cost function -> (1/2m)*sum(Y - Y_hat)^.2

    Args:
        Y: labels of data
        Y_hat: Predictions(activations) from a last layer, the output layer

    Returns:
        cost: The Squared Error Cost result
        dY_hat: gradient of Cost w.r.t the Y_hat

    """
    m = Y.shape[0]

    cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
    cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar

    dY_hat = -1 / m * (Y - Y_hat)  # derivative of the squared error cost function

    return cost, dY_hat


def predict(X, Y, Zs, As):
    """
    helper function to predict on data using a neural net model layers

    Args:
        X: Data in shape (features x num_of_examples)
        Y: labels in shape ( label x num_of_examples)
        Zs: All linear layers in form of a list e.g [Z1,Z2,...,Zn]
        As: All Activation layers in form of a list e.g [A1,A2,...,An]
    Returns::
        p: predicted labels
        probas : raw probabilities
        accuracy: the number of correct predictions from total predictions
    """
    m = X.shape[1]
    n = len(Zs)  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    Zs[0].forward(X)
    As[0].forward(Zs[0].Z)
    for i in range(1, n):
        Zs[i].forward(As[i-1].A)
        As[i].forward(Zs[i].Z)
    probas = As[n-1].A

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:  # 0.5 is threshold
            p[0, i] = probas[0, i] # was 1 for binary funct
        else:
            p[0, i] = probas[0, i] # was 0 for binary funct

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    ## for bunary funct --- >accuracy = np.sum((p == Y) / m)
    rms_error = np.sqrt(np.mean((p-Y)**2))

    return p, probas, rms_error


def plot_learning_curve(costs, learning_rate, total_epochs, save=False):
    """
    This function plots the Learning Curve of the model

    Args:
        costs: list of costs recorded during training
        learning_rate: the learning rate during training
        total_epochs: number of epochs the model was trained for
        save: bool flag to save the image or not. Default False
    """
    # plot the cost
    plt.figure()

    steps = int(total_epochs / (len(costs)))  # the steps at with costs were recorded
    plt.ylabel('Cost')
    plt.xlabel('Iterations ')
    plt.title("Learning rate =" + str(learning_rate))
    plt.plot(np.squeeze(costs))
    locs, labels = plt.xticks()
    plt.xticks(locs[1:-1], tuple(np.array(locs[1:-1], dtype='int')*steps))  # change x labels of the plot
    plt.xticks()
    if save:
        plt.savefig('Cost_Curve.png', bbox_inches='tight')
    plt.show()


