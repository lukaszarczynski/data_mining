x = rand(100, 1);
y = rand(100, 1);
w = rand(100, 1);
X = rand(100, 1000);

l = v_length(x);
a = w_average(x, w);
e = euclid(x, y);
s = scalar_product(x, y);

L = v_length(X);
A = w_average(X, w);
E = euclid(X, y);
S = scalar_product(X, y);
