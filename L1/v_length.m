function [length] = v_length(M)
    v_2 = M .^ 2;
    len = size(M);
    len = len(1);
    length = sqrt(ones(1, len) * v_2);
end
