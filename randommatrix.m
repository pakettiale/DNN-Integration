

xs = linspace(-1,1,100);
plot(xs, custom_tanh(custom_tanh(custom_tanh(custom_tanh(custom_tanh(xs))))))
y = [];
for x = xs
    y(end+1) = net(x);
end
%plot(xs, y)    

function out = net(in)
    W1 = randn(1, 64)/5+0.07;
W2 = randn(64, 64)/5+0.07;
W3 = randn(64, 64)/5+0.07;
W4 = randn(64, 64)/5+0.07;
W5 = randn(64, 1)/5+0.07;
b1 = zeros(1, 64);
b2 = zeros(1, 64);
b3 = zeros(1, 64);
b4 = zeros(1, 64);
b5 = zeros(1,1);

    X2 = tanh(in*W1+b1);
    X3 = tanh(X2*W2+b2);
    X4 = tanh(X3*W3+b3);
    X5 = tanh(X4*W4+b4);
    out  = 1/(1+exp(-(X5*W5+b5)));
end

function out = custom_tanh(in)
    out = tanh(in)*0.3 + 0.7*in;
end
