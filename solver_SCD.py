import sys
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import math

def updateValue(oldValue, w, x, y, c):
    # r = float(y[0][0]);
    a = (y.sum(0)*float((w.dot(x.transpose())).sum(0))); # O(d)
    b = oldValue - (a - 1)/c;
    # print(b);
    return (b if b > 0 else 0) if (b if b > 0 else 0) < 1 else 1;

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
    # For how many iterations do we wish to execute SCD?
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
    
    trainingSet = np.random.permutation(Xtr.shape[0]);

    # The labels are named 1 and 2 in the data set. Convert them to our standard -1 and 1 labels
    Ytr = 2*(Ytr - 1.5);
    Ytr = Ytr.astype(int);
    testSetY = Ytr[trainingSet[240000:300000]];
    trainingSetY = Ytr[trainingSet[0:240000]];

    # Optional: densify the features matrix.
    # Warning: will slow down computations
    Xtr = Xtr.toarray();
    testSetX = Xtr[trainingSet[240000:300000],:];
    trainingSetX = Xtr[trainingSet[0:240000],:];

    # We have n data points each in d-dimensions
    n, d = trainingSetX.shape;

    if (flag == 1): # this is inaccurate only for representation purposes
        w = csr_matrix(np.load("model_SCD.npy"));
        x = csr_matrix(testSetX);
        y = csr_matrix(testSetY);
        # evaluator = x.dot(w.transpose()).multiply(y.transpose()); # X*W'.*Y'
        y_pred = csr_matrix(predict(w, x));
        print(y_pred.shape);
        print(y.shape);
        print("Test Set Accuracy: " + str(evaluate(y_pred, y)));
        exit();

    y = csr_matrix(trainingSetY).transpose();
    x = csr_matrix(trainingSetX);
    Q = x.multiply(x).sum(1); # n * 1 ith row stores x'*x

    # Initialize model
    # For dual SCD, you will need to maintain d_alpha and w
    # Note: if you have densified the Xt matrix then you can initialize w as a NumPy array
    w = csr_matrix((1, d));
    d_alpha = np.random.uniform(low=0.0, high=1.0, size=(n,));
    
    # We will take a timestamp after every "spacing" iterations
    time_elapsed = np.zeros(math.ceil(n_iter/spacing));
    tick_vals = np.zeros(math.ceil(n_iter/spacing));
    obj_val = np.zeros(math.ceil(n_iter/spacing));
    
    tick = 0;
    
    ttot = 0.0;
    t_start = datetime.now();
    
    for t in range(n_iter):		
        ### Doing dual SCD ###
        
        # Choose a random coordinate from 1 to n
        i_rand = random.randint(0,n - 1); # should be 0 to n - 1
        
        # Store the old and compute the new value of alpha along that coordinate
        d_alpha_old = d_alpha[i_rand];
        d_alpha[i_rand] = updateValue(d_alpha_old, w, x.getrow(i_rand), y[i_rand], Q[i_rand]);
        # print(d_alpha[i_rand]);
        # Update the model - takes only O(d) time!
        w = w + (d_alpha[i_rand] - d_alpha_old)*y[i_rand]*x.getrow(i_rand);
        # Take a snapshot after every few iterations
        # Take snapshots after every spacing = 5000 or so SCD iterations since they are fast
        if t%spacing == 0:
            # Stop the timer - we want to take a snapshot
            t_now = datetime.now();
            delta = t_now - t_start;
            time_elapsed[tick] = ttot + delta.total_seconds();
            ttot = time_elapsed[tick];
            tick_vals[tick] = tick;
            evaluator = x.dot(w.transpose()).multiply(y);
            obj_val[tick] = cal(evaluator, w); # Calculate the objective value f(w) for the current model w^t
            print(obj_val[tick]);
            tick = tick+1;
            # Start the timer again - training time!
            t_start = datetime.now();
            
    w_final = w.toarray();
    np.save("model_SCD.npy", w_final);
    
    x = csr_matrix(testSetX);
    y = csr_matrix(testSetY);
    # evaluator = x.dot(w.transpose()).multiply(y.transpose()); # X*W'.*Y'
    y_pred = predict(w, x);
    print("Test Set Accuracy: " + str(evaluate(y_pred, y)));
if __name__ == '__main__':
    main()