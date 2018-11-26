% Machine Learning (20CS6037-001)
% Assignment 3
% Group Name: LI_LI_SONG_ZENG
% Group Members: Haipeng Li, Xin Li, Ximing Song, Jianfeng Zeng


%######################################################################################


% 
% close all; clear all; clc;
% 
% [training_data_set, testing_data_set, data] = makeData();
% 
% [w, b, a] = SMO(data, 0.1, 0.1, 0.5, training_data_set);
% 
% 
% % Accuracy and F-measure
% x=testing_data_set(:,1:end-1);
% y=testing_data_set(:,end);
% 
% fx=sign(w*x'+b)';
% [~, Accuracy, F_measure] = summary(y, fx)
% 
% %Plotting the Decision Boundry
% hold on
% scatter(x(y==1,1),x(y==1,2),'b')
% scatter(x(y==-1,1),x(y==-1,2),'r')
% 
% syms x1 x2
% fn=vpa((w(1)*x1+b)/w(2),6);
% fn1=vpa((w(1)*x1+b-1)/w(2),6);
% fn2=vpa((w(1)*x1+b+1)/w(2),6);
% fplot(fn,'Linewidth',2);
% fplot(fn1,'Linewidth',1);
% fplot(fn2,'Linewidth',1);
% axis([2 10 1 5])
% xlabel ('Positive Class: blue, Negative Class: red')
% hold off




%######################################################################################


close all; clear all; clc;

data=csvread('LinearlySeprerableData.csv');
data(:,1:end-1)=zscore(data(:,1:end-1));

data_size = size(data, 1)

% Split the data set into 2 subsets
seq = randperm(numel(linspace(1, data_size, data_size)));

% Set training data set
training_data_set = data(seq(1:(data_size/2)),:);

% Set testing data set
testing_data_set = data(seq((data_size/2)+1:end),:);

%[w, b, a] = SMO(data, 0.1, 0.1, 0.5, training_data_set);
[w, b, a] = SMO(data, 0.1, 0.1, 0.5, training_data_set);

% Accuracy and F-measure
x=testing_data_set(:,1:end-1);
y=testing_data_set(:,end);

fx=sign(w*x'+b)';
[~, Accuracy, F_measure] = summary(y, fx)

%Plotting the Decision Boundry
hold on
scatter(x(y==1,1),x(y==1,2),'b')
scatter(x(y==-1,1),x(y==-1,2),'r')

syms x1 x2
fn=vpa((w(1)*x1+b)/w(2),6);
fn1=vpa((w(1)*x1+b-1)/w(2),6);
fn2=vpa((w(1)*x1+b+1)/w(2),6);
fplot(fn,'Linewidth',2);
fplot(fn1,'Linewidth',1);
fplot(fn2,'Linewidth',1);

axis([-2 2 -2 2])
xlabel ('Positive Class: blue, Negative Class: red')
hold off



