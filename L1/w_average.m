function [average] = w_average(M, weights)
    len = size(M);
    len = len(1);
    weight_sum = ones(1, len) * weights;
    sum = weights' * M;
    average = sum/weight_sum;
end
