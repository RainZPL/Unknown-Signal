import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import random


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


#split whole array into several parts
def array_split(array, amount):
    #Each part will be assigned 'amount' values
    return [array[i:i+amount] for i in range(0, len(array),amount)]


def ksplit_segment(all_xs,all_ys,k_fold):
    # merge x column and y column together
    merge_xy = np.dstack((all_xs,all_ys))
    # shuffle them to make data dispersion
    np.random.shuffle(merge_xy[0])
    # split the merge_xy into x column and y column to set training and test data
    rand_xs, rand_ys = np.dsplit(merge_xy,2)
    rand_xs, rand_ys = rand_xs[0],rand_ys[0] #delete bracket
    #convert list to float
    rand_xs = list(map(np.float64,rand_xs))
    rand_ys = list(map(np.float64,rand_ys))
    #calculate the size of every part for one segment, and convert to integer
    amount = int(len(rand_xs)/k_fold)
    #split every segment with 4-fold
    data_xs = array_split(rand_xs, amount)
    data_ys = array_split(rand_ys, amount)
    return data_xs,data_ys

def set_train_test(k_fold,data_xs,data_ys):   
    error1 = []
    error2 = []
    error3 = []
    error4 = []
    errorSin = []
    errorExp = []
#     errorLog = []

    # The loop makes each group iterate through as 1 test set 
    # and it ends when all groups have been used as test set
    # will also calculate error
    for i in range(k_fold):
        # set one part data as test set, convert test list to array
        X_test = np.array(data_xs[i])
        Y_test = np.array(data_ys[i])
        X_train = []
        Y_train = []
        # this loop will set other parts as training set
        for ii in range(k_fold):
            if ii != i :
                X_train.extend(data_xs[ii])
                Y_train.extend(data_ys[ii])
        #convert train list to array
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # use training set into least square function to calculate coefficients
        a1, b1 = linear_least_squares(X_train, Y_train)
        a2, b2, c2 = quadratic_least_squares(X_train, Y_train)
        a3, b3, c3, d3 =  cubic_least_squares(X_train,Y_train)
        a4, b4, c4 , d4, e4 =  biquadrate_least_squares(X_train,Y_train)
        sin_a, sin_b = sin_least_squares(X_train,Y_train)
        exp_a, exp_b = exp_least_squares(X_train,Y_train)
#         log_a, log_b = log_least_squares(X_train,Y_train)
        # use test set to calculate y hat
        y_hat1 = a1 + b1 * X_test
        y_hat2 = a2 + b2 * X_test + c2*(X_test**2)
        y_hat3 = a3 + b3 * X_test + c3 * (X_test**2) + d3 * (X_test**3)
        y_hat4 = a4 + b4 * X_test + c4 * (X_test**2) + d4 * (X_test**3) + e4*(X_test**4)
        y_hatSin = sin_a + sin_b * np.sin(X_test)
        y_hatExp = exp_a + exp_b * np.exp(X_test)
#         y_hatLog = log_a + log_b * np.log(X_test)
#         v1 = linear_least_squares(X_train, Y_train)
#         v2 = quadratic_least_squares(X_train, Y_train)
#         v3 =  cubic_least_squares(X_train,Y_train)
#         v4 =  biquadrate_least_squares(X_train,Y_train)
#         v5 = sin_least_squares(X_train,Y_train)
#         y_hat1 = np.polyval(np.flip(v1,0),X_test)
#         y_hat2 = np.polyval(np.flip(v2,0),X_test)
#         y_hat3 = np.polyval(np.flip(v3,0),X_test)
#         y_hat4 = np.polyval(np.flip(v4,0),X_test)
#         y_hatSin = np.polyval(np.flip(v5,0),X_test)

        # calculate current error and append it
        error1.append(square_error(Y_test,y_hat1))
        error2.append(square_error(Y_test,y_hat2))
        error3.append(square_error(Y_test,y_hat3))
        error4.append(square_error(Y_test,y_hat4))
        errorSin.append(square_error(Y_test,y_hatSin))
        errorExp.append(square_error(Y_test,y_hatExp))
#         errorLog.append(square_error(Y_test,y_hatLog))

    # calculate the mean of error
    mean_error1 = np.mean(error1)
    mean_error2 = np.mean(error2)
    mean_error3 = np.mean(error3)
    mean_error4 = np.mean(error4)
    mean_errorSin = np.mean(errorSin)
    mean_errorExp = np.mean(errorExp)
#     mean_errorLog = np.mean(errorLog)

    # determine the minimum error
    typefunc = min(mean_error1,mean_error2,mean_error3,mean_error4,mean_errorSin,mean_errorExp)

    # return function type
    if typefunc == mean_error1 :
        return 1
    # because k-fold cross-validation does not solve overfitting
    # the polynomial function type will dafault as cubic
    elif typefunc == mean_error2 :
        return 3
    elif typefunc == mean_error3 :
        return 3
    elif typefunc == mean_error4 :
        return 3
    elif typefunc == mean_errorSin :
        return 5
    else :
        return 6

    
def linear_least_squares(xs1, ys1):
    # extend the first column with 1s
    ones = np.ones(xs1.shape)
    # merge 1s and x data to make matrix
    x_e = np.column_stack((ones, xs1))
    # use formula for the matrix form of least squares to get matirx A(coefficient)
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys1)
    return v


def quadratic_least_squares(xs1, ys1):
    ones = np.ones(xs1.shape)
    x_e = np.column_stack((ones, xs1,xs1**2))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys1)
    return v

def cubic_least_squares(xs1,ys1):
    ones = np.ones(xs1.shape)
    x_e = np.column_stack((ones,xs1,xs1**2,xs1**3))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys1)
    return v


def biquadrate_least_squares(xs1,ys1):
    ones = np.ones(xs1.shape)
    x_e = np.column_stack((ones,xs1,xs1**2,xs1**3,xs1**4))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys1)
    return v


def sin_least_squares(xs1, ys1):
    ones = np.ones(xs1.shape)
    x_e = np.column_stack((ones, np.sin(xs1)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys1)
    return v

def exp_least_squares(xs1, ys1):
    ones = np.ones(xs1.shape)
    x_e = np.column_stack((ones, np.exp(xs1)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys1)
    return v

# def log_least_squares(xs1, ys1):
#     ones = np.ones(xs1.shape)
#     x_e = np.column_stack((ones, np.log(xs1)))
#     v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys1)
# #     a ,b = v
# #     y_hat =  a + b * np.sin(xtest)
# #     error = square_error(ytest,y_hat)
# #     print(v)
#     return v
    
    
    
# calculate square error
def square_error(y,y_hat):
     return np.sum((y-y_hat)**2)

    

def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour,label="data")
    plt.show()


# xs,ys = load_points_from_file('basic_2.csv')

# load train data
xs,ys = load_points_from_file(sys.argv[1]) 

#split all data segment
all_xs = array_split(xs,20)
all_ys = array_split(ys,20)
#set k_fold value
k_fold = 5
total_error = []
fig, ax = plt.subplots()
# this loop iterates through each segment
for index in range (len(all_xs)):
    # set random seed to make random shuffle regular
    np.random.seed(3)
    # split data with several parts
    data_xs,data_ys= ksplit_segment(all_xs[index],all_ys[index],k_fold)
    # calculate function type
    type_func = set_train_test(k_fold,data_xs,data_ys)
    # set linefit x
    linefit_x = np.linspace(all_xs[index].min(),all_xs[index].max(),1000)
    if type_func == 1 :
        # calculate the error for this segment
        v = linear_least_squares(all_xs[index], all_ys[index])
        y_hat1 = v[0] + v[1] * all_xs[index]
        total_error.append(square_error(all_ys[index],y_hat1))
        # set linefit y
        linefit_y = v[0] + v[1] * linefit_x
        if len(sys.argv) == 3:
            if sys.argv[2]=="--plot":
                # show the fitting line
                ax.plot(linefit_x, linefit_y,'b',label="Segment %d: linear"%(index+1))
#     elif type_func == 2 :
#         v = quadratic_least_squares(all_xs[index], all_ys[index])
#         y_hat2 = v[0] + v[1] * all_xs[index] + v[2] * (all_xs[index]**2)
#         total_error.append(square_error(all_ys[index],y_hat2))
#         linefit_y = v[0] + v[1] * linefit_x  + v[2] * (linefit_x**2)
#         if len(sys.argv) == 3:
#             if sys.argv[2]=="--plot":
#         ax.plot(linefit_x, linefit_y, 'r',label="polynomial")
    elif type_func == 3 :
        v = cubic_least_squares(all_xs[index], all_ys[index])
        y_hat3 = v[0] + v[1] * all_xs[index] + v[2] * (all_xs[index]**2) + v[3]*(all_xs[index]**3)
        total_error.append(square_error(all_ys[index],y_hat3))
        linefit_y = v[0] + v[1] * linefit_x  + v[2] * (linefit_x**2) + v[3]*(linefit_x**3)
        if len(sys.argv) == 3:
            if sys.argv[2]=="--plot":
                ax.plot(linefit_x, linefit_y, 'y', label = "Segment %d: cubic polynomial"%(index+1))
#     elif type_func == 4 :
#         v = biquadrate_least_squares(all_xs[index], all_ys[index])
#         y_hat4 = v[0] + v[1] * all_xs[index] + v[2] * (all_xs[index]**2) + v[3] *(all_xs[index]**3) + v[4] * (all_xs[index]**4)
#         total_error.append(square_error(all_ys[index],y_hat4))
#         linefit_y = v[0] + v[1] * linefit_x  + v[2] * (linefit_x**2) + v[3]*(linefit_x**3) + v[4] * (linefit_x**4)
# #         if len(sys.argv) == 3:
# #             if sys.argv[2]=="--plot":
#         ax.plot(linefit_x, linefit_y, 'black', lw=1)
    else :
        v = sin_least_squares(all_xs[index], all_ys[index])
        y_hatSin = v[0] + v[1] * np.sin(all_xs[index])
        total_error.append(square_error(all_ys[index],y_hatSin))
        linefit_y = v[0] + v[1] * np.sin(linefit_x)
        if len(sys.argv) == 3:
            if sys.argv[2]=="--plot":
                ax.plot(linefit_x, linefit_y, 'purple',label="Segment %d: sin"%(index+1))

 
# sum each segment error and get total error
total_error = np.sum(total_error)
print(total_error)
if len(sys.argv) == 3:
    if sys.argv[2]=="--plot":
        # show the figure
        ax = plt.gca()
        ax.legend(loc = 'best')
        view_data_segments(xs, ys)

        