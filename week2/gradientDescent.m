function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


x = X(:,2);

for iter = 1:num_iters,

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it ca4519.767868n be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    

hypot = X * theta;   
temp_first = theta(1) - alpha*(1/m)*sum((hypot-y).*X(:,1));    
temp_sec = theta(2) - alpha*(1/m)*sum((hypot-y).*X(:,2)); 
theta = [temp_first;temp_sec];    

fprintf('Current theta: %f\n',theta);

    % ============================================================

    % Save the cost J in every iteration    
J_history(iter) = computeCost(X, y, theta);
    
fprintf('Current Cost func: %f\n',J_history(iter));

end


end
