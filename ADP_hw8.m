
n = 0.5;
R = 0.4;
P = @(s, f) normcdf((norminv(s ./ (1-R),0,1) - sqrt(n) .* f) ./ sqrt(1-n),0,1);

a = @(s,f,u) (1-P(s,f)) + P(s,f) .* exp(-1i .* u .* (1-R));

feiL = @(f,u) a(6.8/1000,f,u);

m = @(u) integral(@(f) feiL(f,u) .* 1./ sqrt(2.* pi) .* exp(-f.^2 ./2),-100,100);

hold on;
X = @(u,z) m(u) .* exp(-1i .*u .*z) ./ (1i .*u);
z = 0.1:0.01:1;
Pr =0.5 + 1 ./ pi .* integral(@(u) real(X(u,z)),0,100,'ArrayValued',true);
