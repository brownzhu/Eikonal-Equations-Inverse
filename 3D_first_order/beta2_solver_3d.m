function beta = beta2_solver_3d(T, res, dx, dy, dz)
% BETA2_SOLVER_3D  Solve for normalized adjoint weight beta in 3D.
%
% Interior transport PDE:
%     T_x * beta_x + T_y * beta_y + T_z * beta_z = 0   in Omega
%
% Boundary condition:
%     (n · ∇T) * beta = res                            on ∂Omega
% where res is a residual field (e.g. T^* - T or J_r(T^* - T)).
% Only boundary values of res are used.
%
% Grid convention (same as your codes):
%   - i: y-direction (dy)
%   - j: x-direction (dx)
%   - k: z-direction (dz)
%   - T(i,j,k) at (x_j, y_i, z_k)
%
% Inputs:
%   T   : (I,J,K) traveltime
%   res : (I,J,K) residual field (boundary values used)
%   dx,dy,dz : spacings
%
% Output:
%   beta: (I,J,K)

[I, J, K] = size(T);

% ---------- input checks ----------
if ~isequal(size(res), [I, J, K])
    error("beta2_solver_3d: size(res) must match size(T). Got (%d,%d,%d) vs (%d,%d,%d).", ...
        size(res,1), size(res,2), size(res,3), I, J, K);
end
if dx <= 0 || dy <= 0 || dz <= 0
    error("beta2_solver_3d: dx,dy,dz must be positive.");
end

% ---------- parameters ----------
iterFS = 500;
tolFS  = 1e-10;
epsg   = 1e-12;

inv_dx = 1/dx; inv_dy = 1/dy; inv_dz = 1/dz;

beta = zeros(I, J, K);

%% =========================================================
% 1) Boundary condition: (n · ∇T) * beta = res on ∂Ω
%    res is given directly (only boundary used).
%% =========================================================

% --- outward normal derivatives (one-sided) on 6 faces ---

% x-faces (j direction)
% j=1:  n=(-1,0,0) => n·∇T = -T_x ≈ -(T(:,2,:) - T(:,1,:))/dx
% j=J:  n=(+1,0,0) => n·∇T = +T_x ≈ +(T(:,J,:) - T(:,J-1,:))/dx
gXmin = -(T(:,2,:)   - T(:,1,:))   * inv_dx;     % I×1×K
gXmax =  (T(:,J,:)   - T(:,J-1,:)) * inv_dx;     % I×1×K

% y-faces (i direction)
% i=1:  n=(0,-1,0) => n·∇T = -T_y ≈ -(T(2,:,:) - T(1,:,:))/dy
% i=I:  n=(0,+1,0) => n·∇T = +T_y ≈ +(T(I,:,:) - T(I-1,:,:))/dy
gYmin = -(T(2,:,:)   - T(1,:,:))   * inv_dy;     % 1×J×K
gYmax =  (T(I,:,:)   - T(I-1,:,:)) * inv_dy;     % 1×J×K

% z-faces (k direction)
% k=1:  n=(0,0,-1) => n·∇T = -T_z ≈ -(T(:,:,2) - T(:,:,1))/dz
% k=K:  n=(0,0,+1) => n·∇T = +T_z ≈ +(T(:,:,K) - T(:,:,K-1))/dz
gZmin = -(T(:,:,2)   - T(:,:,1))   * inv_dz;     % I×J×1
gZmax =  (T(:,:,K)   - T(:,:,K-1)) * inv_dz;     % I×J×1

% protect denominators
gXmin = protect_denom(gXmin, epsg);
gXmax = protect_denom(gXmax, epsg);
gYmin = protect_denom(gYmin, epsg);
gYmax = protect_denom(gYmax, epsg);
gZmin = protect_denom(gZmin, epsg);
gZmax = protect_denom(gZmax, epsg);

% --- face beta values ---
beta(:,1,:) = res(:,1,:) ./ gXmin;
beta(:,J,:) = res(:,J,:) ./ gXmax;
beta(1,:,:) = res(1,:,:) ./ gYmin;
beta(I,:,:) = res(I,:,:) ./ gYmax;
beta(:,:,1) = res(:,:,1) ./ gZmin;
beta(:,:,K) = res(:,:,K) ./ gZmax;

% --- edges: average of two adjacent faces ---
beta(1,1,:) = 0.5*( res(1,1,:) ./ gXmin(1,1,:) + res(1,1,:) ./ gYmin(1,1,:) );
beta(I,1,:) = 0.5*( res(I,1,:) ./ gXmin(I,1,:) + res(I,1,:) ./ gYmax(1,1,:) );
beta(1,J,:) = 0.5*( res(1,J,:) ./ gXmax(1,1,:) + res(1,J,:) ./ gYmin(1,J,:) );
beta(I,J,:) = 0.5*( res(I,J,:) ./ gXmax(I,1,:) + res(I,J,:) ./ gYmax(1,J,:) );

beta(1,:,1) = 0.5*( res(1,:,1) ./ gYmin(1,:,1) + res(1,:,1) ./ gZmin(1,:,1) );
beta(I,:,1) = 0.5*( res(I,:,1) ./ gYmax(1,:,1) + res(I,:,1) ./ gZmin(I,:,1) );
beta(1,:,K) = 0.5*( res(1,:,K) ./ gYmin(1,:,K) + res(1,:,K) ./ gZmax(1,:,1) );
beta(I,:,K) = 0.5*( res(I,:,K) ./ gYmax(1,:,K) + res(I,:,K) ./ gZmax(I,:,1) );

beta(:,1,1) = 0.5*( res(:,1,1) ./ gXmin(:,1,1) + res(:,1,1) ./ gZmin(:,1,1) );
beta(:,J,1) = 0.5*( res(:,J,1) ./ gXmax(:,1,1) + res(:,J,1) ./ gZmin(:,J,1) );
beta(:,1,K) = 0.5*( res(:,1,K) ./ gXmin(:,1,K) + res(:,1,K) ./ gZmax(:,1,1) );
beta(:,J,K) = 0.5*( res(:,J,K) ./ gXmax(:,1,K) + res(:,J,K) ./ gZmax(:,J,1) );

% --- corners: average of three faces ---
beta(1,1,1) = ( res(1,1,1)/gXmin(1,1,1) + res(1,1,1)/gYmin(1,1,1) + res(1,1,1)/gZmin(1,1,1) )/3;
beta(I,1,1) = ( res(I,1,1)/gXmin(I,1,1) + res(I,1,1)/gYmax(1,1,1) + res(I,1,1)/gZmin(I,1,1) )/3;
beta(1,J,1) = ( res(1,J,1)/gXmax(1,1,1) + res(1,J,1)/gYmin(1,J,1) + res(1,J,1)/gZmin(1,J,1) )/3;
beta(I,J,1) = ( res(I,J,1)/gXmax(I,1,1) + res(I,J,1)/gYmax(1,J,1) + res(I,J,1)/gZmin(I,J,1) )/3;

beta(1,1,K) = ( res(1,1,K)/gXmin(1,1,K) + res(1,1,K)/gYmin(1,1,K) + res(1,1,K)/gZmax(1,1,1) )/3;
beta(I,1,K) = ( res(I,1,K)/gXmin(I,1,K) + res(I,1,K)/gYmax(1,1,K) + res(I,1,K)/gZmax(I,1,1) )/3;
beta(1,J,K) = ( res(1,J,K)/gXmax(1,1,K) + res(1,J,K)/gYmin(1,J,K) + res(1,J,K)/gZmax(1,J,1) )/3;
beta(I,J,K) = ( res(I,J,K)/gXmax(I,1,K) + res(I,J,K)/gYmax(1,J,K) + res(I,J,K)/gZmax(I,J,1) )/3;

%% =========================================================
% 2) Interior: upwind coefficients for T_x beta_x + T_y beta_y + T_z beta_z = 0
%    (keep it close to your original-style loops for correctness)
%% =========================================================

a_p = zeros(I,J,K); a_m = a_p;   % y
b_p = zeros(I,J,K); b_m = b_p;   % x
c_p = zeros(I,J,K); c_m = c_p;   % z

for i = 2:I-1
    for j = 2:J-1
        for k = 2:K-1
            ay_p = -(T(i+1,j,k) - T(i,j,k)) * inv_dy;
            ay_m = -(T(i,j,k)   - T(i-1,j,k)) * inv_dy;

            ax_p = -(T(i,j+1,k) - T(i,j,k)) * inv_dx;
            ax_m = -(T(i,j,k)   - T(i,j-1,k)) * inv_dx;

            az_p = -(T(i,j,k+1) - T(i,j,k)) * inv_dz;
            az_m = -(T(i,j,k)   - T(i,j,k-1)) * inv_dz;

            a_p(i,j,k) = max(ay_p, 0);  a_m(i,j,k) = min(ay_m, 0);
            b_p(i,j,k) = max(ax_p, 0);  b_m(i,j,k) = min(ax_m, 0);
            c_p(i,j,k) = max(az_p, 0);  c_m(i,j,k) = min(az_m, 0);
        end
    end
end

%% =========================================================
% 3) Fast sweeping (8 directions)
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
    beta_old = beta;
    ord = orders{mod(it-1,8) + 1};

    for i = ord(1):ord(2):ord(3)
        for j = ord(4):ord(5):ord(6)
            for k = ord(7):ord(8):ord(9)

                denom = (a_p(i,j,k) - a_m(i,j,k))*inv_dy ...
                      + (b_p(i,j,k) - b_m(i,j,k))*inv_dx ...
                      + (c_p(i,j,k) - c_m(i,j,k))*inv_dz;

                if abs(denom) < 1e-14
                    continue;
                end

                num = (a_p(i,j,k)*beta(i-1,j,k) - a_m(i,j,k)*beta(i+1,j,k))*inv_dy ...
                    + (b_p(i,j,k)*beta(i,j-1,k) - b_m(i,j,k)*beta(i,j+1,k))*inv_dx ...
                    + (c_p(i,j,k)*beta(i,j,k-1) - c_m(i,j,k)*beta(i,j,k+1))*inv_dz;

                beta(i,j,k) = num / denom;
            end
        end
    end

    err = norm(beta(:) - beta_old(:)) * dx * dy * dz;
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