function [w, b, a] = SMO(data, eps, tol, C, training)
% Input:
%  data: original data (in our case 150x74 double)
%  eps: error rate
%  tol: tolerence
%  C: regularization parameter
%  training: training data set

% Output: 
%  w: weight vector
%  b: the bias
%  a: a vector of lagrange multipliers

% Init data
X=data(:,1:end-1);
Y=data(:,end);
X_train = training(:,1:end-1);
training_size = size(X_train, 1);
X = X(1:training_size, :);
Y = Y(1:training_size);
degree = size(X(1,:));
degree = degree(2);

% Init error
Error=zeros(1, training_size);

% Init weight vector
w = zeros(1, degree); 

% Let K denote a kernel on SXS. K_(ij)=K(x_i,x_j)=x_i*x_j
K = zeros(size(X,1));
for i = 1:training_size
    for j = 1:training_size
        K(i,j) = sum(X(i,:).*X(j,:));
    end
end

% Step 1: initialize a = {a_1,...,a_l} randomly subject to constraint sum(y,a)=0
a = zeros(1, training_size);

% initialize b = 0
b = 0;

% Prepare counters for the while loop
num_of_changed = 0;
check_all = 1;

while (num_of_changed>0) || (check_all==1)

    num_of_changed=0;

    if(check_all)
        
        for i2 = 1:training_size
            [add, a, b, Error, w] = checkExamples(i2, Y, a, tol, C, Error, eps, K, b, w, X_train);
            num_of_changed = num_of_changed+add;
        end
        
    else
        
        % iterate through data and see where alpha is not 0 and not C
        indx = find(Error > tol & Error < C - tol);
        
        for j = indx
            [add, a, b, Error, w] = checkExamples(j, Y, a, tol, C, Error, eps, K, b, w, X_train);
            num_of_changed = num_of_changed + add;
        end
        
    end
    
    if (check_all == 1)
        
        check_all = 0;
        
    elseif (num_of_changed == 0)
        
        check_all = 1;
        
    end
end
