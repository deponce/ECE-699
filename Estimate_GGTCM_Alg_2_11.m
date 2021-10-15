clear all;
clc;
%=======================================================================%
%                                                                       %
%         For the TCM algorithm detail plase refer to the paper         %
%                                                                       %
% Transparent Composite Model for DCT Coefficients: Design and Analysis %
%                                                                       %
%=======================================================================%

%---------------------------- Load data --------------------------------%
%----------------------------CW 11 9063---------------------------------%
position_x = 1;
position_y = 1;
pcnt_inliers = 0.95;
myFolder = './';
if ~isfolder(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder);
    uiwait(warndlg(errorMessage));
    myFolder = uigetdir(); % Ask for a new one.
    if myFolder == 0
         % User clicked Cancel
         return;
    end
end

DCT_cof_pos = [];
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    load(fullFileName);
    DCT_cof_pos = [DCT_cof_pos; DCT_cof(:,position_x,position_y)];
    if k >= 1
        break
    end
end
fprintf('load Done!\n');
DCT_cof_pos= reshape(DCT_cof_pos,1,[]);
Y = DCT_cof_pos;

%-------------------------- init para --------------------------------%

abs_DCT_cof_pos = abs(DCT_cof_pos);
sorted_abs_DCT_cof_pos = sort(abs_DCT_cof_pos);
d = sorted_abs_DCT_cof_pos(round(size(sorted_abs_DCT_cof_pos,2)*pcnt_inliers));
%d = sorted_abs_DCT_cof_pos(end-1);
fprintf('d: %g \n',d);
% d=0.24;

a = get_max_DCT_value(8/255, position_x, position_y);
a = max(DCT_cof_pos);
% position = (1,1);
%--------------------------   GGTCM   --------------------------------%
fprintf('start GGTCM!\n');

[counts, centers] = hist(Y, 99);
bar(centers, counts);
hold on
[current_alpha, current_beta, current_yc, current_b,current_G] = GGTCM_estimate(Y, a, d);

GGTCM_pdf = @(data, alpha, beta, y_c, b, a)...
            b*beta./(2.*alpha.*gamma(1/beta).*gammainc((y_c/alpha).^beta, 1/beta,'lower')).*exp(-1*(abs(data)./alpha).^beta).*(abs(data)<y_c)+... 
            (1-b)/(2*(a-y_c)).*(abs(data)>y_c);
        
y_hat = GGTCM_pdf(centers, current_alpha, current_beta, current_yc, current_b, a);

y_hat = y_hat./sum(abs(y_hat)).*sum(abs(counts));

plot(centers, y_hat);


function [cnt] = get_max_DCT_value(max_val, u, v)
    u = u-1; v = v-1;
    cnt = 0;
    for x = 0:7
        for y = 0:7
            cnt = cnt + abs(cos((2*x+1)*u*pi/16)*cos((2*y+1)*v*pi/16));
        end
    end
    cnt = 1/8*cnt*max_val;
end 


function [current_alpha,current_beta,current_yc,current_b,current_G] = GGTCM_estimate(Y_n, a, d) %max_DCT_table
    current_G = -Inf;
    current_alpha = 0; current_beta = 0;current_yc = 0;current_b = 0;
    mean0_abs_arr = abs(Y_n);
    fprintf('start sort\n');
    Y_n = sort(Y_n);
    mean0_abs_arr = sort(mean0_abs_arr);
    fprintf('finish sort\n');
    split_lst = unique(mean0_abs_arr(mean0_abs_arr>=d));
    split_lst = sort(split_lst);
    n_sample = size(Y_n,2);
    m = 1; % ?
    n = size(split_lst,2);
    Alpha = 0.01;
    Beta = 0.14;
    
    function [G, Alpha, Beta, B] = calc_g(data, N_1, N1_p, n_sample, alpha, beta, alpha_p, beta_p, y_c, a)
        t = (y_c/alpha)^beta;
        t_p = (y_c/alpha_p)^beta_p;
        b_p = N1_p/n_sample;
        b_n = N_1/n_sample;
        N_yc = N1_p - N_1;
        g_yc_in_n1 = log(b_p*beta_p/(2*alpha_p*gamma(1/beta_p)*gammainc(t_p, 1/beta_p,'lower')))-N_yc*t_p;
        g_yc_in_n2 = 0;
        if a ~= y_c && b_n<1
            g_yc_in_n2 = log((1-b_n)/(2*(a-y_c)));
        end
        if g_yc_in_n1>g_yc_in_n2
            g_N1 = N1_p*log(b_p*beta_p/(2*alpha_p*gamma(1/beta_p)*gammainc(t_p, 1/beta_p,'lower')))-sum((data(:,1:N1_p)./alpha_p).^beta_p);
            g_N2 = 0;
            if a ~= y_c && b_p<1
                g_N2 = (n_sample-N1_p)*log((1-b_p)/(2*(a-y_c)));
            end
            g_N3 = N_yc*g_yc_in_n1;
            Alpha = alpha_p;
            Beta = beta_p;
            B = b_p;
        else
            g_N1 = N_1*log(b_n*beta/(2*alpha*gamma(1/beta)*gammainc(t, 1/beta,'lower')))-sum((data(:,1:N_1)./alpha).^beta);
            g_N2 = 0;
            if a ~= y_c
                g_N2 = (n_sample-N_1)*log((1-b_n)/(2*(a-y_c)));
            end
            g_N3 = N_yc*g_yc_in_n2;
            Alpha = alpha;
            Beta = beta;
            B = b_n;
        end
        G = g_N1 + g_N2 + g_N3;
    end

    for idx = m:n
        fprintf('%g/%g\n',idx-m+1,n-m+1);
        y_c = split_lst(idx);
        yc_lst = find(mean0_abs_arr==y_c);
        N1 = yc_lst(1)-1;
        N1_P = yc_lst(end);
        data_p = mean0_abs_arr(mean0_abs_arr<=y_c);
        data_n = mean0_abs_arr(mean0_abs_arr<y_c);
        
        GGTCM_N1 = @(data_n, alpha, beta)...
            beta./(2.*alpha.*gamma(1/beta).*gammainc((y_c/alpha).^beta, 1/beta,'lower')).*exp(-1*(data_n./alpha).^beta);
        
        GGTCM_N1_p = @(data_p, alpha, beta)...
            beta./(2.*alpha.*gamma(1/beta).*gammainc((y_c/alpha).^beta, 1/beta,'lower')).*exp(-1*(data_p./alpha).^beta);
        
        pd_p = mle(data_p,'pdf' ,GGTCM_N1_p ,'Start',[Alpha, Beta], 'LowerBound',[0.00000001, 0.0000001],'UpperBound',[1, 2]);
        
        pd_n = mle(data_n,'pdf' ,GGTCM_N1 ,'Start',[Alpha, Beta], 'LowerBound',[0.00000001, 0.0000001],'UpperBound',[1, 2]);
        
        [G, Alpha, Beta, B] = calc_g(mean0_abs_arr, N1, N1_P, n_sample, pd_n(1), pd_n(2), pd_p(1), pd_p(2), y_c, a);

         if G > current_G
           current_G = G;
           current_alpha = Alpha;
           current_beta = Beta;
           current_yc = y_c;
           current_b = B;
         end
    end
end

%G 3081622