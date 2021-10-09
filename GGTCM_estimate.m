clear all;
%filename = './ECE-699/FGSM_EPS8/mat_data/FGSM_EPS850.mat';
%load './ECE-699/FGSM_EPS8/mat_data/FGSM_EPS850.mat'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
position_x = 1;
position_y = 1;


myFolder = './ECE-699/FGSM_EPS8/mat_data';
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
end
fprintf('load Done!');
DCT_cof_pos= reshape(DCT_cof_pos,1,[]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = DCT_cof_pos;




GGD = @(data, alpha, beta, mu) ...
    beta/(2*alpha*gamma(1/beta))*exp(-(abs(data-mu)/alpha).^beta);

GGTCM = @(data, alpha, beta, yc)...
    (sum(abs(data)<yc)./size(data,1)).*beta./ ...
    (2.*alpha.*gammainc((yc/alpha).^beta, 1/beta)).*exp(-1*(abs(data)./alpha).^beta) .*(abs(data)<yc) + ... 
    (1-sum(abs(data)<yc)./size(data,1))./(2.*(max(data)-yc)) .*(yc<abs(data)) + ...
    max( ...
    (sum(abs(data)<yc)./size(data,1)).*beta./ ...
    (2.*alpha.*gammainc((yc/alpha).^beta,1/beta)).*exp(-(abs(data)./alpha).^beta), ...
    (1-sum(abs(data)<yc)./size(data,1))./(2.*(max(data)-yc)) ...
    ) .*(abs(data)==yc);


%pd = mle(data,'pdf' ,GGD ,'Start',[0.0618, 1.4982, 0.00000], 'LowerBound',[0, 0, -Inf]);
%pd am()
pd = mle(data,'pdf' ,GGTCM ,'Start',[0.0618, 1.4982, 0.2], 'LowerBound',[0.05, 1.3, 0.1],'UpperBound',[0.1, 2, 0.24]);
h = histogram(data, 200);
y_hat = GGTCM(h.BinEdges, pd(1), pd(2), pd(3))
%fprintf(y_hat)
hold on
plot(h.BinEdges, GGTCM(h.BinEdges, pd(1), pd(2), pd(3)))

