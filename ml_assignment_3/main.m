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

fn=vpa((w(1)*x1+b)/w(2),6);
fn1=vpa((w(1)*x1+b-1)/w(2),6);
fn2=vpa((w(1)*x1+b+1)/w(2),6);

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







%######################################################################################

% 
% close all; clear all; clc;
% 
% data=csvread('./LinearlySeprerableData/LinearlySeprerableData.csv');
% data(:,1:end-1)=zscore(data(:,1:end-1));
% 
% data_size = size(data, 1)
% 
% % % Split the data set into 2 subsets
% % seq = randperm(numel(linspace(1, data_size, data_size)));
% % 
% % % Set training data set
% % training_data_set = data(seq(1:(data_size/2)),:);
% % 
% % % Set testing data set
% % testing_data_set = data(seq((data_size/2)+1:end),:);
% 
% 
% ratio = 0.8;
% 
% % Set training data set
% training_data_set = data(1:(data_size*ratio),:);
% 
% % Set testing data set
% testing_data_set = data((data_size*ratio):end,:);
% 
% [w, b, a] = SMO(data, 0.1, 0.1, 0.5, training_data_set);
% 
% 
% 
% 
% 
% 
% %Plotting the Decision Boundry
% x=training_data_set(:,1:end-1);
% y=training_data_set(:,end);
% 
% 
% % % Weights
% % w=sum(a.*y*x)
% % % Bias
% % b =mean(y - x*w')
% 
% 
% 
% hold on
% scatter(x(y==1,1),x(y==1,2),'b')
% scatter(x(y==-1,1),x(y==-1,2),'r')
% 
% syms x1 x2
% 
% % fn=vpa((w(1)*x1+b)/w(2),6);
% % fn1=vpa((w(1)*x1+b-1)/w(2),6);
% % fn2=vpa((w(1)*x1+b+1)/w(2),6);
% 
% fn=vpa((-b-w(1)*x1)/w(2),6);
% fn1=vpa((-1-b-w(1)*x1)/w(2),6);
% fn2=vpa((1-b-w(1)*x1)/w(2),6);
% 
% fplot(fn,'Linewidth',2);
% fplot(fn1,'Linewidth',1);
% fplot(fn2,'Linewidth',1);
% 
% axis([-2 2 -2 2])
% xlabel ('Positive Class: blue, Negative Class: red')
% hold off
% 
% % %----------------------------------------------------- test summary
% 
% % Accuracy and F-measure
% x_test=testing_data_set(:,1:end-1);
% y_test=testing_data_set(:,end);
% 
% fx=sign(w*x_test'+b)';
% [~, Accuracy, F_measure] = summary(y_test, fx)
% 



%######################################################################################

% 
% close all; clear all; clc;
% 
% %[training_data_set, testing_data_set, data] = makeData();
% 
% data=csvread('./2d_data/2d_dataset.csv');
% training_data_set=csvread('./2d_data/2d_dataset_training.csv');
% testing_data_set=csvread('./2d_data/2d_dataset_testing.csv');
% 
% data_size = size(data, 1)
% 
% [w, b, a] = SMO(data, 0.1, 0.1, 0.5, training_data_set);
% 
% 
% %----------------------------------------------------- plot training
% x=training_data_set(:,1:end-1);
% y=training_data_set(:,end);
% 
% 
% 
% 
% 
% % Weights
% w=sum(a.*y*x)
% % Bias
% b =mean(y - x*w')
% 
% 
% %Plotting the Decision Boundry
% hold on
% scatter(x(y==1,1),x(y==1,2),'b')
% scatter(x(y==-1,1),x(y==-1,2),'r')
% 
% syms x1 x2
% 
% % fn=vpa((w(1)*x1+b)/w(2),6);
% % fn1=vpa((w(1)*x1+b-1)/w(2),6);
% % fn2=vpa((w(1)*x1+b+1)/w(2),6);
% 
% fn=vpa((-b-w(1)*x1)/w(2),6);
% fn1=vpa((-1-b-w(1)*x1)/w(2),6);
% fn2=vpa((1-b-w(1)*x1)/w(2),6);
% 
% 
% fplot(fn,'Linewidth',2);
% fplot(fn1,'Linewidth',1);
% fplot(fn2,'Linewidth',1);
% %axis([0 7 0 7])
% %axis([-10 10 -10 10])
% axis([-20 20 -20 20])
% xlabel ('Positive Class: blue, Negative Class: red')
% hold off
% 
% %----------------------------------------------------- test summary
% 
% % Accuracy and F-measure
% xtest=testing_data_set(:,1:end-1);
% ytest=testing_data_set(:,end);
% 
% fx=sign(w*xtest'+b)';
% [~, Accuracy, F_measure] = summary(ytest, fx)
% 
% 
% 






