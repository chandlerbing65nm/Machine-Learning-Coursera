function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z=0;
for i = 1:size(X,2)
    f = theta(i).*X(:,i);
    z = z + f;
end

h = 1./(1+exp(-z));
sigma = y.*log(h)+(1-y).*log(1-h);

J = (-1/m)*sum(sigma) + (lambda/(2*m))*sum(theta([2:1:size(X, 2)],:).^2);

for i = 1:size(X, 2)
    if i == 1
        grad(i) = (1/m)*sum((h-y).*X(:,i));
    else
        grad(i) = (1/m)*sum((h-y).*X(:,i)) + (lambda/m).*theta(i,:);
    end
end

% =============================================================

end
