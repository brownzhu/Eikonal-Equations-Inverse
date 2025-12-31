function F = odd_fft(f, dim, n)
% Perform DST-I along dimension dim using odd extension

sz = size(f);
N = 2*(n+1);

% move dim to first
perm = [dim, setdiff(1:ndims(f), dim)];
f = permute(f, perm);

% reshape to 2D for convenience
f = reshape(f, n, []);

% odd extension
f_ext = zeros(N, size(f,2));
f_ext(2:n+1, :) = f;
f_ext(n+3:end, :) = -flipud(f);

% FFT
F_ext = fft(f_ext, [], 1);

% extract sine modes
F = imag(F_ext(2:n+1, :));

% restore shape
F = reshape(F, sz(perm));
F = ipermute(F, perm);
end
