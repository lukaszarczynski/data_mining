rng('default')
uniform_distribution = -1 + 2 * rand(10000, 1);
hist(uniform_distribution, 100)
x = -1:0.01:1;
uniform_pdf = arrayfun(@(x) 0.5, x);
plot(x, uniform_pdf)
uniform_distribution_2 = -1 + 2 * rand(100000, 1);
hist(uniform_distribution_2, 100)

normal_distribution = 3 .* randn(10000, 1) + 5;
hist(normal_distribution, 100)
x2 = -10:0.1:20;
normal_pdf = (1/(3*sqrt((2*pi))) * exp(-0.5*((x2-5)/3).^2));
plot(x2, normal_pdf)
normal_distribution_2 = 3 .* randn(100000, 1) + 5;
hist(normal_distribution_2, 100)

normal_x = sqrt(5) .* randn(10000, 1) + 2;
normal_y = 1 .* randn(10000, 1) + 3;
plot(normal_x, normal_y, 'r.')
axis([-6 10 -1 7])

normal_x_2 = sqrt(5) .* randn(100000, 1) + 2;
normal_y_2 = 1 .* randn(100000, 1) + 3;
plot(normal_x_2, normal_y_2, 'r.')
axis([-6 10 -1 7])

mu = [2; 3];
sigma = [5 0; 0 1];
[x_3, y_3] = meshgrid(-6:.4:10, -1:.2:7);
const = 1/(2*pi*sqrt(det(sigma)));
temp = [x_3(:) - mu(1) y_3(:) - mu(2)];
pdf = const * exp(-0.5*diag(temp*inv(sigma)*temp'));
pdf = reshape(pdf, size(x_3));
surf(x_3, y_3, pdf)
axis([-6 10 -1 7])

probability = sum(normal_x < normal_y) / 10000;
probability_2 = sum(normal_x_2 < normal_y_2) / 100000;

% P(X < Y) = P(X - Y < 0) = 1 - P(X - Y >= 0)
% X - Y = Norm(2, 5) - Norm(3, 1) = Norm(2-3, 5+1)
% 1 - P(Norm(-1, 6) >= 0) = 1 - Phi(1/sqrt(6)) = 1 - 0.3409 = 0.6591
