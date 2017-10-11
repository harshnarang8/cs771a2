import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import math
from tqdm import tqdm

def h(n):
    return 1;

def cal(evaluator, w):
    temp = csr_matrix(np.ones(evaluator.shape)) - evaluator; # 1 - evaluator
    z = csr_matrix((temp > 0).toarray().astype(int));
    t = w.dot(w.transpose()).multiply(0.5);
    # print((t + temp.multiply(z).sum(0))[0][0]); #0.5*||w||^2 + [1 - evaluator]_+
    return float((t + temp.multiply(z).sum(0))[0][0]);

def evaluate(y_pred, y):
    temp = (y_pred == y).toarray().astype(int);
    totalSize = temp.shape[1];
    correctSize = temp.sum();
    print(correctSize)
    print(totalSize);
    return correctSize/totalSize;

def predict(w, x):
    return x.dot(w.transpose()).sign().transpose();

def main():

    # Get training file name from the command line
    traindatafile = sys.argv[1];
    # For how many iterations do we wish to execute GD?
    n_iter = int(sys.argv[2]);
    # After how many iterations do we want to timestamp?
    spacing = int(sys.argv[3]);
    # Do we want to evaluate the saved model?
    if (len(sys.argv) > 4):
        flag = int(sys.argv[4]);
    else:
        flag = 0;
    
    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile);

    Xtr = tr_data[0]; # Training features in sparse format
    Ytr = tr_data[1]; # Training labels
    
    # We have n data points each in d-dimensions
    n, d = Xtr.get_shape();
    
    # The labels are named 1 and 2 in the data set. Convert them to our standard -1 and 1 labels
    Ytr = 2*(Ytr - 1.5);
    Ytr = Ytr.astype(int);
    
    # Optional: densify the features matrix.
    # Warning: will slow down computations
    Xtr = Xtr.toarray();

    # trainingSet = np.random.permutation(Xtr.shape[0]);
    trainingSet = np.arange(Xtr.shape[0]);
    print(trainingSet.shape)
    testSetX = Xtr[trainingSet[240000:300000],:];
    testSetY = Ytr[trainingSet[240000:300000]];
    trainingSetX = Xtr[trainingSet[0:240000],:];
    trainingSetY = Ytr[trainingSet[0:240000]];

    if (flag == 1): # this is inaccurate only for representation purposes
        w = csr_matrix(np.load("model_GD.npy"));
        x = csr_matrix(Xtr);
        y = csr_matrix(Ytr);
        # evaluator = x.dot(w.transpose()).multiply(y.transpose()); # X*W'.*Y'
        y_pred = csr_matrix(predict(w, x));
        print(y_pred.shape);
        print(y.shape);
        print("Test Set Accuracy: " + str(evaluate(y_pred, y)));
        exit();

    if (flag == 2): # this is inaccurate only for representation purposes
        w = csr_matrix(np.load("model_GD.npy"));
        x = csr_matrix(testSetX);
        y = csr_matrix(testSetY);
        # evaluator = x.dot(w.transpose()).multiply(y.transpose()); # X*W'.*Y'
        y_pred = csr_matrix(predict(w, x));
        print(y_pred.shape);
        print(y.shape);
        print("Test Set Accuracy: " + str(evaluate(y_pred, y)));
        exit();

    # Initialize model
    # For primal GD, you only need to maintain w
    # Note: if you have densified the Xt matrix then you can initialize w as a NumPy array
    w = csr_matrix((1, d));
    wbar = csr_matrix((1, d));
    x = csr_matrix(trainingSetX);
    y = csr_matrix(trainingSetY);
    G = 0;
    
    # We will take a timestamp after every "spacing" iterations
    time_elapsed = np.zeros(math.ceil(n_iter/spacing));
    tick_vals = np.zeros(math.ceil(n_iter/spacing));
    obj_val = np.zeros(math.ceil(n_iter/spacing));
    
    tick = 0;
    
    ttot = 0.0;
    t_start = datetime.now();
    
    temp = 0;
    count = 0;
    for t in tqdm(range(n_iter)):
        ### Doing primal GD ###
        # Compute gradient
        v = csr_matrix((1, d));
        evaluator = x.dot(w.transpose()).multiply(y.transpose()); # X*W'.*Y'
        # print(evaluator.shape);
        
        # for j in tqdm(range(n)):
        #     if (evaluator[j] > 1):
        #         evaluator[j] = 0;
        #     else:
        #         evaluator[j] = -1;
        # the following line does the job of the above commented loop
        temp1 = csr_matrix((evaluator < 1).toarray().astype(int)).multiply(-1);
        # print(temp1);
        # print("H");
        v = temp1.multiply(y.transpose()).transpose().dot(x); # (temp1.*(Y'))'*X

        # print(v.shape);
        # for j in tqdm(range(n)): # adding contribution from each example
        #     temp = f(csr_matrix(Xtr[j]), Ytr[j], w, d);
        #     v = v + temp;
        #     if (j% 3000 == 0):
        #         print(j/3000);
        g = w + v;
        # print(g.shape);
        # Reshaping not required since g is made to be a row vector
        # g.reshape(1,d); # Reshaping since model is a row vector
        G += float(g.dot(g.transpose()).sum(0));
        # Calculate step lenght. Step length may depend on n and t
        # eta = h(n) * 1/math.sqrt(t + 1);
        eta = h(n) * 1/math.sqrt(G + 1e-6); # trying out adagrad
        # eta = 0.00001; # Trying a constant length
        # Update the model
        w = w - eta * g;
        
        # Use the averaged model if that works better (see [\textbf{SSBD}] section 14.3)
        wbar = (wbar*(t) + w)/(t+1);
        # Take a snapshot after every few iterations
        # Take snapshots after every spacing = 5 or 10 GD iterations since they are slow
        if t%spacing == 0:
            # Stop the timer - we want to take a snapshot
            t_now = datetime.now();
            delta = t_now - t_start;
            time_elapsed[tick] = ttot + delta.total_seconds();
            ttot = time_elapsed[tick];
            tick_vals[tick] = tick;
            obj_val[tick] = cal(evaluator, w); # Calculate the objective value f(w) for the current model w^t or the current averaged model \bar{w}^t
            evaluator1 = x.dot(wbar.transpose()).multiply(y.transpose());
            print(str(obj_val[tick]) + " " + str(cal(evaluator1, wbar)));
            if (temp < obj_val[tick]):
                count += 1;
            temp = obj_val[tick];
            tick = tick+1;
            # Start the timer again - training time!
            t_start = datetime.now();
    print("Value of count: " + str(count));
    # Choose one of the two based on whichever works better for you
    w_final = w.toarray();
    # w_final = wbar.toarray();
    np.save("model_GD.npy", w_final);

    x = csr_matrix(testSetX);
    y = csr_matrix(testSetY);
    # evaluator = x.dot(w.transpose()).multiply(y.transpose()); # X*W'.*Y'
    y_pred = predict(w, x);
    print("Test Set Accuracy on w: " + str(evaluate(y_pred, y)));

    y_pred = predict(wbar, x);
    print("Test Set Accuracy on wbar: " + str(evaluate(y_pred, y)));
        
if __name__ == '__main__':
    main()