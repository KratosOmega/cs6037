function [training_data_set, testing_data_set, data] = d2Dataset()

initial_class1 = [1, 1];
initial_class2 = [2, 2];
n = 100;

class1 = zeros(n, 2);
class2 = zeros(n, 2);
positive = zeros(n, 1);
negative = zeros(n, 1);
positive(:,1) = 1;
negative(:,1) = -1;

class1(:,1) = initial_class1(1);
class1(:,2) = initial_class1(2);
class2(:,1) = initial_class2(1);
class2(:,2) = initial_class2(2);

min = 0.5;
max = 1.5;

class1_d1_variance=min+rand(1,n)*(max-min);
class1_d2_variance=min+rand(1,n)*(max-min);
class2_d1_variance=min+rand(1,n)*(max-min);
class2_d2_variance=min+rand(1,n)*(max-min);

class1(:,1) = class1(:,1) + class1_d1_variance';
class1(:,2) = class1(:,2) + class1_d2_variance';
class2(:,1) = class2(:,1) + class2_d1_variance';
class2(:,2) = class2(:,2) + class2_d2_variance';

class1 = [class1 positive];
class2 = [class2 negative];

data = [class1; class2];

shape = size(data);
data_size = shape(1);

% set train/test ratio
subRatio = 0.8;

% Split the data set into 2 subsets
seq = randperm(numel(linspace(1, data_size, data_size)));

% Set training data set
training_data_set = data(seq(1:(data_size*subRatio)),:)

% Set testing data set
testing_data_set = data(seq((data_size*subRatio)+1:end),:)

csvwrite('./2d_data/2d_dataset.csv',data)
csvwrite('./2d_data/2d_dataset_training.csv',training_data_set)
csvwrite('./2d_data/2d_dataset_testing.csv',testing_data_set)

