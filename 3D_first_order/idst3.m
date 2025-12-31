function u = idst3(fh)
% inverse 3D DST-I

[nx, ny, nz] = size(fh);

u = dst3(fh);

u = u / (2*(nx+1)) / (2*(ny+1)) / (2*(nz+1));
end
