function [distances] = fast_matrix_euclid(M1, M2)
    squares_1 = M1 .^ 2;
    squares_2 = M2 .^ 2;

    distances = -2 * M1' * M2;

    sum_squares_1 = sum(squares_1);
    sum_squares_2 = sum(squares_2);

    distances = bsxfun(@plus, distances, sum_squares_1');
    distances = bsxfun(@plus, distances, sum_squares_2);

    distances = sqrt(distances);
