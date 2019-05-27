function [scalar] = scalar_product(M, v2)
    len = size(M);
    count = len(2);
    len = len(1);
    v = M .* repmat(v2, [1, count]);
    scalar = ones(1, len) * v;
end
