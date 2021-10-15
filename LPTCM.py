import numpy as np
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = './'
listdir = os.listdir(data_dir)
pos_arr = np.array([])
Pos_x = 0
Pos_y = 0
pcnt_inliers = 0.90
for name in tqdm(iter(listdir)):

    if name[-3:] != 'npy':
        continue
    print(name)
    #DCT_cof = np.load(data_dir+name)
    DCT_cof = np.load(data_dir+ name)
    DC_DCT_cof = DCT_cof[:,Pos_x,Pos_y]
    pos_arr = np.concatenate((pos_arr,DC_DCT_cof), axis=0)

print(len(pos_arr))
plt.hist(pos_arr,100)
plt.show()
def LPTCM_est(Y_n, a, d):
    g_lst = []
    def determine_theta(Y_n_p, N_1, y_c, last_Lambda):# 6. Determine theta^+(y_c) and theta^-(y_c) to be the ML estimate of theta for truncated distribution
        # Algorithm 2 Estimating Lambda for a Truncated Laplacian model
        p_converge = False
        n_converge = False
        eps = 10**np.floor(math.log10(last_Lambda))/10000 # eps >0 is a small prescribed threshold
        C_n = np.mean(Y_n_p[:N_1])
        C_p = np.mean(Y_n_p)
        if C_p == 0:
            p_converge = True
            Lambda_p = 0
        elif C_p >= y_c/2:
            p_converge = True
            Lambda_p = float('inf')
        else:
            last_Lambda_p = C_p

        if C_n == 0:
            n_converge = True
            Lambda_n = 0
        elif C_n >= y_c/2:
            n_converge = True
            Lambda_n = float('inf')
        else:
            last_Lambda_n = C_n
        cnt_loop = 0
        while cnt_loop<=10000000:
            cnt_loop += 1
            if not p_converge:
                exp_p = math.exp(y_c/last_Lambda_p)
                Lambda_p= C_p+ (y_c)/(exp_p-1)
                if abs(Lambda_p-last_Lambda_p) < eps:
                    p_converge = True
                last_Lambda_p = Lambda_p
            if not n_converge:
                exp_n = math.exp(y_c/last_Lambda_n)
                Lambda_n= C_n+ (y_c)/(exp_n-1)
                if abs(Lambda_n-last_Lambda_n) < eps:
                    n_converge = True
                last_Lambda_n = Lambda_n
            if p_converge and n_converge:
                break
        return Lambda_n, Lambda_p
    current_g = -float('inf')
    select_b = 0
    select_Lambda = 0
    select_yc = 0
    def calc_g(mean0_abs_arr, N_1, N1_p, n_sample, Lambda_n, Lambda_p, y_c):
        # a: the largest magnitude a sample y can take
        b_n = N_1 / n_sample
        b_p = N1_p/n_sample
        N_yc = N1_p - N_1
        g_yc_in_n1 = N_yc*math.log(b_p/(1-math.exp(-y_c/Lambda_p))/(2*Lambda_p))-np.sum(mean0_abs_arr[N_1:N1_p])/Lambda_p

        g_yc_in_n2 = 0
        if a!=y_c and b_n<1:
            g_yc_in_n2 = N_yc*math.log((1-b_n)/(2*(a-y_c)))
        if g_yc_in_n1 > g_yc_in_n2:
            g_N1 = N1_p*math.log(b_p/(1-math.exp(-y_c/Lambda_p))/(2*Lambda_p))-np.sum(mean0_abs_arr[:N_1])/Lambda_p
            g_N2 = 0
            if b_p < 1 and a!=y_c:
                g_N2 = (n_sample-N1_p)*math.log((1-b_p)/(2*(a-y_c)))
            g_nc = g_yc_in_n1
            b = b_p
            Lambda = Lambda_p
        else:
            g_N1 = N1*math.log(b_n/(1-math.exp(-y_c/Lambda_n))/(2*Lambda_n))-np.sum(mean0_abs_arr[:N_1])/Lambda_n
            g_N2 = 0
            if b_n < 1 and a!=y_c:
                g_N2 = (n_sample-N1)*math.log((1-b_n)/(2*(a-y_c)))
            g_nc = g_yc_in_n2
            b = b_n
            Lambda = Lambda_n
        return g_N1+g_N2+g_nc, b, Lambda

    last_Lambda = 1e-4
    mean0_abs_arr = np.abs(Y_n)
    print('start sort')
    Y_n.sort()
    mean0_abs_arr.sort() # 1. sort {|Y_n|} in ascending order into W_1 <=...<=W_n
    print('finish sort')
    #split_lst = np.unique(Y_n[Y_n>=0])
    split_lst = np.unique(mean0_abs_arr)
    split_lst.sort()
    n_sample = len(Y_n)
    m = np.searchsorted(split_lst, d, side='right')-1 # 2. determine m = min{i: W_i>=d}
    n = len(split_lst)
    for idx in tqdm(range(m,n)):
        y_c = split_lst[idx]
        N1 = np.searchsorted(mean0_abs_arr, y_c, side='left')   # Number of points in N1 //regard y_c as outliers
        N1_p = N1+ np.searchsorted(mean0_abs_arr[N1:], y_c, side='right')   # Number of points in N1 //regard y_c as inliers

        Lambda_n, Lambda_p  = determine_theta(mean0_abs_arr[:N1_p], N1, y_c, last_Lambda)

        g, B, Lambda = calc_g(mean0_abs_arr, N1, N1_p, n_sample, Lambda_n, Lambda_p, y_c)
        last_Lambda = Lambda
        g_lst.append(g)
        if g > current_g:
            current_g = g
            select_b = B
            select_Lambda = Lambda
            select_yc = y_c
    return select_b, select_Lambda,select_yc, g_lst

def get_max_DCT_value(max_val=1, u=0, v=0):
    cnt = 0
    A_i = math.sqrt(2/8)
    A_j = math.sqrt(2/8)
    if u == 0:
        A_i = math.sqrt(1/8)
    if v == 0:
        A_j = math.sqrt(1/8)
    for x in range(8):
        for y in range(8):
            cnt += abs(math.cos((2*x+1)*u*math.pi/16.)*math.cos((2*y+1)*v*math.pi/16.))
    return A_i*A_j*cnt*max_val

#a = max_DCT_table[7,7]#0.125490196078431
#a = get_max_DCT_value(8/255, Pos_x, Pos_y)
a = np.max(np.abs(pos_arr))
print('a: ',a)
tmp = np.abs(pos_arr)
tmp.sort()
d= tmp[int(pcnt_inliers*(len(tmp)))]
print('d: ',d)
b, Lambda, yc, g_lst =LPTCM_est(pos_arr,a,d)
print('b: ',b, '\nlambda: ',Lambda, '\nyc: ', yc)
g_lst = np.array(g_lst)
g_lst_diff = g_lst-np.roll(g_lst,1)
plt.plot(g_lst_diff[1:-1])
plt.show()
#plt.hist(pos_arr, 100)


