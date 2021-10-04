import numpy as np
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
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

print("load_data")
def LPTCM(Y_n, a, d):
    def determine_theta(N1, Y_n, y_c):
        eps = 1e-3
        # 6. Determine theta^+(y_c) and theta^-(y_c) to be the ML estimate
        # of theta for truncated distribution
        C = np.mean(np.abs(Y_n[N1]))
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

    def calc_g(N1, N2, Y_n, a, b, y_c, Lambda):
        # a: the largest magnitude a sample y can take
        return np.sum(N1)*math.log(b/(1-math.exp(-y_c/Lambda))*1/(2*Lambda))+np.sum(-np.abs(Y_n[N1])) + np.sum(N2)*(1-b)/(2*(a-y_c))

    b_lst = np.array([])
    Lambda_lst = np.array([])
    g_lst = np.array([])
    print('start sort')
    Y_n.sort() # 1. sort {|Y_n|} in ascending order into W_1 <=...<=W_n
    W_lst = np.unique(Y_n)
    print('finish sort')
    m = np.sum([W_lst<=d]) # 2. determine m = min{i: W_i>=d}
    n = len(W_lst)
    zero_idx = np.sum([Y_n<0])
    idx_choose = 0
    for idx in tqdm(range(m,n)): # 3. foreach y_c in {d, W_m, W_{m+1},...,W_n}do
        y_c = W_lst[idx]
        N1_p, N1 = np.abs(Y_n)<=y_c, np.abs(Y_n)<y_c # 4. set N_1^+(y_c) = {i: |Y_i|<=y_c}, N_1(y_c) = {i: |Y_i|<y_c}
        N2_p, N2 = np.abs(Y_n)>y_c, np.abs(Y_n)>=y_c
        B_p, B_n = np.sum(N1_p)/len(Y_n), np.sum(N1)/len(Y_n) # 5.Compute b^+(y_c) and b^-(y_c)
        Lambda_p, Lambda_n = determine_theta(N1_p, Y_n, y_c), determine_theta(N1, Y_n, y_c) # 6.
        g_p, g_n = calc_g(N1_p, N2_p, Y_n, a, B_p, y_c, Lambda_p), calc_g(N1, N2, Y_n, a, B_n, y_c, Lambda_n)
        if g_p >= g_n:
            print(Lambda_p,g_p)
            g_lst = np.append(g_lst,g_p)
            Lambda_lst = np.append(Lambda_lst,Lambda_p)
            b_lst = np.append(b_lst,B_p)
        else:
            print(Lambda_n,g_n)
            g_lst = np.append(g_lst,g_n)
            Lambda_lst = np.append(Lambda_lst,Lambda_n)
            b_lst = np.append(b_lst,B_n)
    max_idx = np.argmax(g_lst)
    return b_lst[max_idx], Lambda_lst[max_idx]

print(LPTCM(pos_arr, a = 0.061, d= 0.017))
