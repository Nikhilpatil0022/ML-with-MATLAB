function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

%  variables to return  
J = 0;
grad = zeros(size(theta));

%  The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
% Each row of the resulting matrix will contain the value of the prediction for that example.



J =(-1/m) * sum(y.* log(sigmoid(sum(((theta') .* X),2))) + (1-y).*log(1 - sigmoid(sum(((theta') .* X),2)))) + (lambda/(2*m))* sum((theta(2:length(theta),1)).^2);

grad(1) = (1/m) * (sigmoid(sum(((theta') .* X),2)) - y)' * X(:,1);

n = length(grad);
grad(2:n) = (1/m) * (sigmoid(sum(((theta') .* X),2)) - y)' * X(:,2:n) + (lambda/m)*(theta(2:n,:))';


grad = grad(:);

end
