function beta = beta_solver_3d_fast(T, res, dx, dy, dz, squareoption)
% BETA_SOLVER_3D_FAST
% Vectorized acceleration version of beta solver in 3D.
%
% - Boundary initialization uses only 6 face arrays (smaller memory).
% - Upwind coefficient construction is fully vectorized (removes 3 nested loops).
% - Fast sweeping update remains triple loops (Gauss–Seidel dependency).
%
% SPEEDUP (no change in algorithm):
%   1) Remove beta_old copy + full norm scan by accumulating diff^2 on the fly.
%   2) Loop order changed to k->j->i for better memory locality (MATLAB column-major).

[I, J, K] = size(T);
if ~isequal(size(res), [I, J, K])
    error('beta_solver_3d_fast: size(res) must match size(T).');
end

iterFS = 500;
tolFS  = 1e-10;
epsg   = 1e-12;

inv_dx = 1/dx; inv_dy = 1/dy; inv_dz = 1/dz;

beta = zeros(I, J, K);

%% =========================================================
% 1) Boundary initialization: (n·∇T) * beta = res on ∂Ω
%% =========================================================

% j=1 / j=J faces  (y-direction in your comment; but indices are consistent with your code)
gL  = -(T(:,2,:) - T(:,1,:))   * inv_dy;   % I×1×K
gR  =  (T(:,J,:) - T(:,J-1,:)) * inv_dy;   % I×1×K

% i=1 / i=I faces
gT  = -(T(2,:,:) - T(1,:,:))   * inv_dx;   % 1×J×K
gB  =  (T(I,:,:) - T(I-1,:,:)) * inv_dx;   % 1×J×K

% k=1 / k=K faces
gF  = -(T(:,:,2) - T(:,:,1))   * inv_dz;   % I×J×1
gBk =  (T(:,:,K) - T(:,:,K-1)) * inv_dz;   % I×J×1

% protect denominators
gL  = protect_denom(gL,  epsg);
gR  = protect_denom(gR,  epsg);
gT  = protect_denom(gT,  epsg);
gB  = protect_denom(gB,  epsg);
gF  = protect_denom(gF,  epsg);
gBk = protect_denom(gBk, epsg);

% Face values
beta(:,1,:)   = res(:,1,:) ./ gL;
beta(:,J,:)   = res(:,J,:) ./ gR;
beta(1,:,:)   = res(1,:,:) ./ gT;
beta(I,:,:)   = res(I,:,:) ./ gB;
beta(:,:,1)   = res(:,:,1) ./ gF;
beta(:,:,K)   = res(:,:,K) ./ gBk;

% Edges (kept exactly as your original)
beta(1,1,:) = 0.5*( res(1,1,:) ./ gL(1,1,:) + res(1,1,:) ./ gT(1,1,:) );
beta(I,1,:) = 0.5*( res(I,1,:) ./ gL(I,1,:) + res(I,1,:) ./ gB(1,1,:) );
beta(1,J,:) = 0.5*( res(1,J,:) ./ gR(1,1,:) + res(1,J,:) ./ gT(1,J,:) );
beta(I,J,:) = 0.5*( res(I,J,:) ./ gR(I,1,:) + res(I,J,:) ./ gB(1,J,:) );

beta(1,:,1) = 0.5*( res(1,:,1) ./ gT(1,:,1) + res(1,:,1) ./ gF(1,:,1) );
beta(I,:,1) = 0.5*( res(I,:,1) ./ gB(1,:,1) + res(I,:,1) ./ gF(I,:,1) );
beta(1,:,K) = 0.5*( res(1,:,K) ./ gT(1,:,K) + res(1,:,K) ./ gBk(1,:,1) );
beta(I,:,K) = 0.5*( res(I,:,K) ./ gB(1,:,K) + res(I,:,K) ./ gBk(I,:,1) );

beta(:,1,1) = 0.5*( res(:,1,1) ./ gL(:,1,1) + res(:,1,1) ./ gF(:,1,1) );
beta(:,J,1) = 0.5*( res(:,J,1) ./ gR(:,1,1) + res(:,J,1) ./ gF(:,J,1) );
beta(:,1,K) = 0.5*( res(:,1,K) ./ gL(:,1,K) + res(:,1,K) ./ gBk(:,1,1) );
beta(:,J,K) = 0.5*( res(:,J,K) ./ gR(:,1,K) + res(:,J,K) ./ gBk(:,J,1) );

% Corners (kept exactly as your original)
beta(1,1,1) = ( res(1,1,1)/gL(1,1,1) + res(1,1,1)/gT(1,1,1) + res(1,1,1)/gF(1,1,1) )/3;
beta(I,1,1) = ( res(I,1,1)/gL(I,1,1) + res(I,1,1)/gB(1,1,1) + res(I,1,1)/gF(I,1,1) )/3;
beta(1,J,1) = ( res(1,J,1)/gR(1,1,1) + res(1,J,1)/gT(1,J,1) + res(1,J,1)/gF(1,J,1) )/3;
beta(I,J,1) = ( res(I,J,1)/gR(I,1,1) + res(I,J,1)/gB(1,J,1) + res(I,J,1)/gF(I,J,1) )/3;

beta(1,1,K) = ( res(1,1,K)/gL(1,1,K) + res(1,1,K)/gT(1,1,K) + res(1,1,K)/gBk(1,1,1) )/3;
beta(I,1,K) = ( res(I,1,K)/gL(I,1,K) + res(I,1,K)/gB(1,1,K) + res(I,1,K)/gBk(I,1,1) )/3;
beta(1,J,K) = ( res(1,J,K)/gR(1,1,K) + res(1,J,K)/gT(1,J,K) + res(1,J,K)/gBk(1,J,1) )/3;
beta(I,J,K) = ( res(I,J,K)/gR(I,1,K) + res(I,J,K)/gB(1,J,K) + res(I,J,K)/gBk(I,J,1) )/3;

%% =========================================================
% 2) Upwind coefficients (vectorized)
%% =========================================================
Tcoef = T;
if squareoption == 1
    Tcoef = Tcoef.^2;
end

Tx_f = -(Tcoef(2:I,:,: ) - Tcoef(1:I-1,:,:)) * inv_dx;   % (I-1,J,K)
Ty_f = -(Tcoef(:,2:J,: ) - Tcoef(:,1:J-1,:)) * inv_dy;   % (I,J-1,K)
Tz_f = -(Tcoef(:,:,2:K ) - Tcoef(:,:,1:K-1)) * inv_dz;   % (I,J,K-1)

ii = 2:I-1; jj = 2:J-1; kk = 2:K-1;

Tx_p = Tx_f(ii,   jj,   kk);
Tx_m = Tx_f(ii-1, jj,   kk);

Ty_p = Ty_f(ii,   jj,   kk);
Ty_m = Ty_f(ii,   jj-1, kk);

Tz_p = Tz_f(ii,   jj,   kk);
Tz_m = Tz_f(ii,   jj,   kk-1);

a_p = zeros(I,J,K); a_m = zeros(I,J,K);
b_p = zeros(I,J,K); b_m = zeros(I,J,K);
c_p = zeros(I,J,K); c_m = zeros(I,J,K);

a_p(ii,jj,kk) = max(Tx_p, 0);
a_m(ii,jj,kk) = min(Tx_m, 0);

b_p(ii,jj,kk) = max(Ty_p, 0);
b_m(ii,jj,kk) = min(Ty_m, 0);

c_p(ii,jj,kk) = max(Tz_p, 0);
c_m(ii,jj,kk) = min(Tz_m, 0);

%% =========================================================
% 3) Fast sweeping (Gauss–Seidel) — algorithm unchanged
%    SPEEDUP: no beta_old copy; accumulate diff^2 on the fly
%    SPEEDUP: loop order k->j->i (better memory locality)
%% =========================================================
orders = {
    [ 2  1 I-1   2  1 J-1   2  1 K-1 ]
    [ 2  1 I-1   2  1 J-1   K-1 -1 2 ]
    [ 2  1 I-1   J-1 -1 2   2  1 K-1 ]
    [ I-1 -1 2   2  1 J-1   2  1 K-1 ]
    [ I-1 -1 2   J-1 -1 2   K-1 -1 2 ]
    [ 2  1 I-1   J-1 -1 2   K-1 -1 2 ]
    [ I-1 -1 2   2  1 J-1   K-1 -1 2 ]
    [ I-1 -1 2   J-1 -1 2   2  1 K-1 ]
};

for it = 1:iterFS
    ord = orders{mod(it-1,8) + 1};

    i_list = ord(1):ord(2):ord(3);
    j_list = ord(4):ord(5):ord(6);
    k_list = ord(7):ord(8):ord(9);

    diff2 = 0.0;

    for k = k_list
        for j = j_list
            for i = i_list

                denom = (a_p(i,j,k) - a_m(i,j,k))*inv_dx ...
                      + (b_p(i,j,k) - b_m(i,j,k))*inv_dy ...
                      + (c_p(i,j,k) - c_m(i,j,k))*inv_dz;

                if abs(denom) < 1e-14
                    continue;
                end

                num = (a_p(i,j,k)*beta(i-1,j,k) - a_m(i,j,k)*beta(i+1,j,k))*inv_dx ...
                    + (b_p(i,j,k)*beta(i,j-1,k) - b_m(i,j,k)*beta(i,j+1,k))*inv_dy ...
                    + (c_p(i,j,k)*beta(i,j,k-1) - c_m(i,j,k)*beta(i,j,k+1))*inv_dz;

                old = beta(i,j,k);
                new = num / denom;
                beta(i,j,k) = new;

                d = new - old;
                diff2 = diff2 + d*d;
            end
        end
    end

    err = sqrt(diff2) * dx * dy * dz;   % == norm(beta(:)-beta_old(:))*dx*dy*dz
    if err < tolFS
        break;
    end
end

end

%% =========================================================
function g = protect_denom(g, epsg)
% Replace near-zero denominators by signed eps to avoid division by 0.
mask = abs(g) < epsg;
if any(mask(:))
    s = sign(g(mask));
    s(s==0) = 1;
    g(mask) = s .* epsg;
end
end