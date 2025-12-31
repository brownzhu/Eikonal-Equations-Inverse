function u = poisson3d_fd_fullbc(f, bc, h, niu)
% Solve -Δu + alpha*u = f
% Solve  -1/alphaΔu + u = f/alpha
% Solve  -niuΔu + u = f*niu
% Dirichlet BC given on six faces
alpha=1/niu;
f=f/niu;
[nx,ny,nz] = size(f);
N = nx*ny*nz;

idx = @(i,j,k) i + (j-1)*nx + (k-1)*nx*ny;

A   = spalloc(N, N, 7*N);
rhs = zeros(N,1);

for k = 1:nz
for j = 1:ny
for i = 1:nx

    p = idx(i,j,k);

    % center
    A(p,p) = 6/h^2 + alpha;
    rhs(p) = f(i,j,k);

    % x-
    if i > 1
        A(p,idx(i-1,j,k)) = -1/h^2;
    else
        rhs(p) = rhs(p) + bc(1, j+1, k+1)/h^2;
    end

    % x+
    if i < nx
        A(p,idx(i+1,j,k)) = -1/h^2;
    else
        rhs(p) = rhs(p) + bc(nx+2, j+1, k+1)/h^2;
    end

    % y-
    if j > 1
        A(p,idx(i,j-1,k)) = -1/h^2;
    else
        rhs(p) = rhs(p) + bc(i+1, 1, k+1)/h^2;
    end

    % y+
    if j < ny
        A(p,idx(i,j+1,k)) = -1/h^2;
    else
        rhs(p) = rhs(p) + bc(i+1, ny+2, k+1)/h^2;
    end

    % z-
    if k > 1
        A(p,idx(i,j,k-1)) = -1/h^2;
    else
        rhs(p) = rhs(p) + bc(i+1, j+1, 1)/h^2;
    end

    % z+
    if k < nz
        A(p,idx(i,j,k+1)) = -1/h^2;
    else
        rhs(p) = rhs(p) + bc(i+1, j+1, nz+2)/h^2;
    end

end
end
end

u_vec = A \ rhs;

% put solution back into full grid (optional but useful)
u = bc;  % copy boundary
u(2:nx+1,2:ny+1,2:nz+1) = reshape(u_vec,[nx,ny,nz]);

end
