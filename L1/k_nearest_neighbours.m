function [indices] = k_nearest_neighbours(M1, M2, k)
    count = size(M1);
    len = count(1);
    count1 = count(2);
    count2 = size(M2);
    count2 = count2(2);
    
    distances = matrix_euclid(M2, M1);
    
    [~, sorted_indices] = sort(distances);
    indices = sorted_indices(1:k, :);
end
