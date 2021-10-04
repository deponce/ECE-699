import numpy as np
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = './ECE-699/FGSM_EPS8/'
listdir = os.listdir(data_dir)
pos_arr = np.array([])
cnt = 0
for name in tqdm(iter(listdir)):
    cnt +=1
    if name[-3:] != 'npy':
        continue
    DCT_cof = np.load(data_dir+name)
    DC_DCT_cof = DCT_cof[:,7,7]
    pos_arr = np.concatenate((pos_arr,DC_DCT_cof), axis=0)

def LPTCM(Y_n, a, d):
    def determine_theta(Y_n, y_c):
        eps = 1e-4
        # 6. Determine theta^+(y_c) and theta^-(y_c) to be the ML estimate
        # of theta for truncated distribution
        C = np.mean(np.abs(Y_n))
        if C == 0:
            return 0
        elif C >= y_c/2:
            return float('inf')
        else:
            last_Lambda = C
        while True:
            exp = math.exp(y_c/last_Lambda)
            Lambda= C+ (y_c)/(exp-1)
            if abs(Lambda-last_Lambda) < eps:
                break
            last_Lambda = Lambda
        return Lambda
    b_lst = np.array([])
    Lambda_lst = np.array([])
    g_lst = np.array([])
    yc_lst = np.array([])
    def calc_g(mean0_abs_arr, N1_p, n_sample, Lambda, y_c ):
        # a: the largest magnitude a sample y can take
        b = N1_p/n_sample
        N1_log_likelihood = N1_p*math.log(b/(1-math.exp(-y_c/Lambda))/(2*Lambda))-np.sum(mean0_abs_arr[:N1_p])
        N2_log_likelihood = (n_sample-N1_p)*math.log((1-b)/(2*(a-y_c)))
        return N1_log_likelihood + N2_log_likelihood, b

    mean = np.mean(Y_n)
    neg_Y_n = Y_n[Y_n<mean]-mean
    pos_Y_n = Y_n[Y_n>=mean]-mean
    inv_neg_Y_n = -1*(neg_Y_n)
    mean0_abs_arr = np.concatenate((pos_Y_n,inv_neg_Y_n)) # abs(zero_mean(array))
    print('start sort')
    mean0_abs_arr.sort() # 1. sort {|Y_n|} in ascending order into W_1 <=...<=W_n
    print('finish sort')
    split_lst = np.unique(Y_n[Y_n>=mean])-mean
    n_sample = len(Y_n)
    m = np.searchsorted(split_lst, d, side='right') # 2. determine m = min{i: W_i>=d}
    n = len(split_lst)
    i = 0
    for idx in tqdm(range(m,n-1)):
        y_c = split_lst[idx]
        N1_p = np.searchsorted(mean0_abs_arr, y_c, side='right')
        Lambda_p = determine_theta(mean0_abs_arr[:N1_p], y_c)
        g_p, B_p = calc_g(mean0_abs_arr, N1_p, n_sample, Lambda_p, y_c)

        g_lst = np.append(g_lst, g_p)
        b_lst = np.append(b_lst, B_p)
        Lambda_lst = np.append(Lambda_lst,Lambda_p)
        yc_lst = np.append(yc_lst,y_c)
    max_idx = np.argmax(g_lst)
    return b_lst[max_idx], mean, Lambda_lst[max_idx],yc_lst[max_idx]

a = np.max(np.abs(pos_arr))

d= 0.017
print(LPTCM(pos_arr,a,d))
