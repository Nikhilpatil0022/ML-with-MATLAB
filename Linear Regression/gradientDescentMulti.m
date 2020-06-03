function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp = zeros(size(X,2),1);
for iter = 1:num_iters

    for i = 1:size(X,2)
      temp(i,1) = theta(i,1) - alpha * 1/m * sum((X * theta - y).*X(:,i));
    endfor
    theta = temp;

    J_history(iter) = computeCostMulti(X, y, theta);

end

end
