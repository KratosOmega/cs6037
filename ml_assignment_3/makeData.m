function [training_data_set, testing_data_set, data] = makeData()

% Load Fisher's iris data for training
load fisheriris;

% Set data_size to be resued
data_size = size(meas, 1);

% Init labels
labels = ones(data_size, 1);
    
% Set data to iris data
data = [meas, labels];

% Set virginica and versicolor to be -1 and leave setosa to be 1
for i = 51:150
    data(i,5) = -1;
end

% Split the data set into 2 subsets
seq = randperm(numel(linspace(1, data_size, data_size)));

% Set training data set
training_data_set = data(seq(1:(data_size/2)),:);

% Set testing data set
testing_data_set = data(seq((data_size/2)+1:end),:);

% Col 1: Sepal Length
% Col 2: Sepal Width
% Col 3: Petal Length
% Col 4: Petal Width
% Col 5: Label (1: Setosa, -1: not Setosa)


