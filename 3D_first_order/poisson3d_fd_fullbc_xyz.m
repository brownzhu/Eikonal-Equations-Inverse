function u = poisson3d_fd_fullbc_xyz(f, bc, dx, dy, dz, niu)
%POISSON3D_FD_FULLBC_XYZ_KRON  Solve (-niu*Δu + u = f) with Dirichlet BC (full bc grid).
%
% Unknowns are interior nodes only: size(f) = (nx,ny,nz)
% bc is full grid: size(bc) = (nx+2,ny+2,nz+2), boundary values prescribed.
%
% Returns u as full grid (nx+2,ny+2,nz+2), with boundary copied from bc.

    if niu <= 0, error('niu must be positive.'); end
    [nx, ny, nz] = size(f);
    if ~isequal(size(bc), [nx+2, ny+2, nz+2])
        error('bc must have size (nx+2, ny+2, nz+2) matching f.');
    end

    % ---------- coefficients ----------
    cx = niu / dx^2;
    cy = niu / dy^2;
    cz = niu / dz^2;

    % ---------- build (or reuse) sparse matrix A ----------
    % A = -niu * (Dxx + Dyy + Dzz) + I  on interior unknowns
    persistent cache
    key = [nx, ny, nz, dx, dy, dz, niu];  % simple cache key (double compare ok if identical params)
    A = [];

    if ~isempty(cache)
        % try find exact match
        hit = find(arrayfun(@(s) isequal(s.key, key), cache), 1);
        if ~isempty(hit)
            A = cache(hit).A;
        end
    end

    if isempty(A)
        ex = ones(nx,1);
        ey = ones(ny,1);
        ez = ones(nz,1);

        % 1D second derivative matrices (Dirichlet handled via rhs, so interior-only stencil)
        Lx = spdiags([ex -2*ex ex], [-1 0 1], nx, nx) / dx^2;
        Ly = spdiags([ey -2*ey ey], [-1 0 1], ny, ny) / dy^2;
        Lz = spdiags([ez -2*ez ez], [-1 0 1], nz, nz) / dz^2;

        Ix = speye(nx); Iy = speye(ny); Iz = speye(nz);

        % 3D Laplacian on interior unknowns (ordering: i fastest, then j, then k)
        Lap = kron(Iz, kron(Iy, Lx)) + kron(Iz, kron(Ly, Ix)) + kron(Lz, kron(Iy, Ix));

        N = nx*ny*nz;
        A = speye(N) - niu * Lap;   % because Lap already is (u_{i-1}-2u_i+u_{i+1})/h^2

        % store cache (keep small)
        entry.key = key;
        entry.A   = A;
        if isempty(cache)
            cache = entry;
        else
            cache(end+1) = entry; %#ok<AGROW>
            if numel(cache) > 5  % avoid blowing memory
                cache = cache(end-4:end);
            end
        end
    end

    % ---------- build rhs with vectorized boundary contributions ----------
    rhs3 = f;  % (nx,ny,nz)

    % x-min face touches bc(1, j+1, k+1)
    rhs3(1,:,:)   = rhs3(1,:,:)   + cx * bc(1,      2:ny+1, 2:nz+1);
    % x-max face touches bc(nx+2, j+1, k+1)
    rhs3(end,:,:) = rhs3(end,:,:) + cx * bc(nx+2,   2:ny+1, 2:nz+1);

    % y-min face touches bc(i+1, 1, k+1)
    rhs3(:,1,:)   = rhs3(:,1,:)   + cy * bc(2:nx+1, 1,      2:nz+1);
    % y-max face touches bc(i+1, ny+2, k+1)
    rhs3(:,end,:) = rhs3(:,end,:) + cy * bc(2:nx+1, ny+2,   2:nz+1);

    % z-min face touches bc(i+1, j+1, 1)
    rhs3(:,:,1)   = rhs3(:,:,1)   + cz * bc(2:nx+1, 2:ny+1, 1);
    % z-max face touches bc(i+1, j+1, nz+2)
    rhs3(:,:,end) = rhs3(:,:,end) + cz * bc(2:nx+1, 2:ny+1, nz+2);

    rhs = rhs3(:);

    % ---------- solve ----------
    u_vec = A \ rhs;

    % ---------- put back to full grid ----------
    u = bc;
    u(2:nx+1, 2:ny+1, 2:nz+1) = reshape(u_vec, [nx, ny, nz]);
end