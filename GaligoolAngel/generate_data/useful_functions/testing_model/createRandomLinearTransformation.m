function [Y, A] = createRandomLinearTransformation(X)
% ---- Description ----
% A function that takes a set of measurements from X and creates a random
% linear transformation A such that Y = AX
% ---- Inputs ----
% X - a matrix, number of parameteres x number of measurements
% ---- Outputs ----
% Y - the X data after the linear transformation. A matrix number of
% parameters x number of measurements
% A - The liner transformation. a square matrix the size of number of
% parameters x the number fo parameters.

    A = rand(size(X,1));
    Y = A * X;
end
