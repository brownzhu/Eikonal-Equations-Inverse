function fh = dst3(f)
% 3D DST-I via odd extension + FFT
% f: nx x ny x nz

[nx, ny, nz] = size(f);

% ---- x direction ----
Fx = odd_fft(f, 1, nx);

% ---- y direction ----
Fxy = odd_fft(Fx, 2, ny);

% ---- z direction ----
Fxyz = odd_fft(Fxy, 3, nz);

fh = Fxyz;
end
