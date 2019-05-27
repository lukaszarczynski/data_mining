function [distances] = matrix_euclid(M1, M2)
    count = size(M1);
    len = count(1);
    count1 = count(2);
    count2 = size(M2);
    count2 = count2(2);
    
    squares_1 = M1 .^ 2;
    sum_squares_1 = ones(1, len) * squares_1;
    sum_squares_1 = repmat(sum_squares_1', 1, count2);
    
    squares_2 = M2 .^ 2;
    sum_squares_2 = ones(1, len) * squares_2;
    sum_squares_2 = repmat(sum_squares_2, count1, 1);
    
    product_2ab = 2 * M1' * M2;
    
    distances = sum_squares_1 + sum_squares_2 - product_2ab;
    distances = sqrt(distances);
end
