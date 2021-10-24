clear all;
clc;
position_x = 5;
position_y = 5;
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
fprintf('load Done!');

data_n = DCT_cof_pos;
%data_n = DCT_cof_pos(1:100000);

[counts, centers] = hist(data_n, 500);
counts = counts/sum(counts);
bar(centers, counts);
hold on
GG = @(data, alpha, beta)...
            beta./(2.*alpha.*gamma(1/beta)).*exp(-1*(abs(data)./alpha).^beta);
Alpha = 0.01;
Beta = 0.12;
pd = mle(data_n,'pdf' ,GG ,'Start',[Alpha, Beta], 'LowerBound',[0.00000001, 0.0000001],'UpperBound',[1, 2]);


y_hat = GG(centers,pd(1) ,pd(2));
y_hat = y_hat/sum(y_hat);
plot(centers, y_hat);


split_lst = unique(data_n(data_n>0));
split_lst = sort(split_lst);

split_lst_size = size(split_lst, 2);

G = sum(log(GG(data_n, pd(1) ,pd(2))))
%5.024861473781907e+06