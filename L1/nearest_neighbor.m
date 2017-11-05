function [indices] = nearest_neighbour(M1, M2)
    count = size(M1);
    len = count(1);
    count1 = count(2);
    count2 = size(M2);
    count2 = count2(2);
    
    distances = matrix_euclid(M2, M1);
    
    [~, indices] = min(distances);
end
