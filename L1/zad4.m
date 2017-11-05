X = rand(100, 1000);
Y = rand(100, 1000);

X_2 = rand(100, 10000);
Y_2 = Y;

tic
D_1 = matrix_euclid(X, Y);
toc
tic
D_2 = matrix_euclid(X_2, Y_2);
toc
