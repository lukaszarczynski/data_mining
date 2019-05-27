function [distance] = euclid(M1, v2)
    count = size(M1);
    count = count(2);
    v = M1 - repmat(v2, [1, count]);
    distance = v_length(v);
end
