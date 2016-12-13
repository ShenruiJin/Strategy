
a = sqrt(0.114625);
S = 1.0613;
r = 0.0048;
d = -0.00819;
T = 7/360;
K = 1.07294055;
H = 1;

Fidt = (log(S) - log(H) + (r - d + 1/2 .* a.^2) .* T) ./ (a .* sqrt(T));
Second = (log(H) - log(S) + (r -d + 1/2 .*a .^2) .* T) ./ ( a .* sqrt(T));

cdfPi1 = normcdf(Fidt) - ( S ./ H ) .^ (1- 2* (r-d) ./ (a .^2)) * normcdf(Second);
cdfPi2 = normcdf(Fidt - a .* sqrt(T)) - ...
    ( S ./ H ) .^ (1- 2* (r-d) ./ (a .^2)) * normcdf(Second - ...
    a .* sqrt(T));

	
Result = exp(-d .* T) * S * cdfPi1 - cdfPi2 * exp(-r .* T) * K;