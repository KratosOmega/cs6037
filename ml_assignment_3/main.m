% Machine Learning (20CS6037-001)
% Assignment 3
% Group Name: LI_LI_SONG_ZENG
% Group Members: Haipeng Li, Xin Li, Ximing Song, Jianfeng Zeng


%######################################################################################


close all; clear all; clc;

%[training_data_set, testing_data_set, data] = makeData();

data=csvread('./iris_data/iris_data.csv');
training_data_set=csvread('./iris_data/iris_training_data_set.csv');
testing_data_set=csvread('./iris_data/iris_testing_data_set.csv');

data_size = size(data, 1)

[w, b, a] = SMO(data, 0.1, 0.1, 0.5, training_data_set);


%----------------------------------------------------- plot training
x=training_data_set(:,1:end-1);
y=training_data_set(:,end);



% Weights
w=sum(a.*y*x)
% Bias
b =mean(y - x*w')


%Plotting the Decision Boundry
hold on
scatter(x(y==1,1),x(y==1,2),'b')
scatter(x(y==-1,1),x(y==-1,2),'r')

syms x1 x2

% fn=vpa((-b-w(1)*x1)/w(2),6);
% fn1=vpa((-1-b-w(1)*x1)/w(2),6);
% fn2=vpa((1-b-w(1)*x1)/w(2),6);

fplot(fn,'Linewidth',2);
fplot(fn1,'Linewidth',1);
fplot(fn2,'Linewidth',1);
%axis([2 10 1 5])
%axis([-10 10 -10 10])
axis([-20 20 -20 20])
xlabel ('Positive Class: blue, Negative Class: red')
hold off

%----------------------------------------------------- test summary

% Accuracy and F-measure
xtest=testing_data_set(:,1:end-1);
ytest=testing_data_set(:,end);

fx=sign(w*xtest'+b)';
[~, Accuracy, F_measure] = summary(ytest, fx)











