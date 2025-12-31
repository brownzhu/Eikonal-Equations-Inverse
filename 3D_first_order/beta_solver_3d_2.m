function beta = beta_solver_3d_2(T, res, dx, dy, dz, squareoption)
% BETA_SOLVER_3D_NDGRID
% Solve the adjoint weight beta in 3D using upwind finite differences + fast sweeping.
%
% PDE (interior):
%   T_x * beta_x + T_y * beta_y + T_z * beta_z = 0     in Omega
%
% Boundary condition (six faces):
%   (n · grad T) * beta = res                          on ∂Omega
% where res = T^* - T is given (boundary values are used).
%
% Grid / index convention (ndgrid style, consistent with T(i,j,k)):
%   i -> x-direction, spacing dx
%   j -> y-direction, spacing dy
%   k -> z-direction, spacing dz
%
% Inputs:
%   T            : traveltime field, size (I,J,K)
%   res          : residual field, res = T_star - T, same size as T
%                 (only boundary values are needed; interior may be unused)
%   dx, dy, dz   : grid spacings
%   squareoption : if 1, use T := T.^2 in transport coefficients (your option)
%
% Output:
%   beta         : adjoint weight, size (I,J,K)

[I, J, K] = size(T);
if ~isequal(size(res), [I, J, K])
    error('beta_solver_3d_ndgrid: size(res) must match size(T).');
end

iterFS = 500;
tolFS  = 1e-10;
epsg   = 1e-12;

beta = zeros(I, J, K);

%% =========================================================
% Boundary normal derivatives (six faces)
% BC: (n·∇T) beta = res  on each face
%% =========================================================

% We only need denominators on faces; keep arrays for simplicity.
gL  = zeros(I,J,K);   % y-face at j=1  (left)
gR  = zeros(I,J,K);   % y-face at j=J  (right)
gT  = zeros(I,J,K);   % x-face at i=1  (top/front in x)
gB  = zeros(I,J,K);   % x-face at i=I  (bottom/back in x)
gF  = zeros(I,J,K);   % z-face at k=1  (front in z)
gBk = zeros(I,J,K);   % z-face at k=K  (back in z)

% Faces with outward normal:
% j=1:   n = (0,-1,0) => n·∇T = -T_y ≈ -(T(:,2,:)-T(:,1,:))/dy
% j=J:   n = (0,+1,0) => n·∇T = +T_y ≈ +(T(:,J,:)-T(:,J-1,:))/dy
gL(:,1,:) = -(T(:,2,:) - T(:,1,:))   / dy;
gR(:,J,:) =  (T(:,J,:) - T(:,J-1,:)) / dy;

% i=1:   n = (-1,0,0) => n·∇T = -T_x ≈ -(T(2,:,:)-T(1,:,:))/dx
% i=I:   n = (+1,0,0) => n·∇T = +T_x ≈ +(T(I,:,:)-T(I-1,:,:))/dx
gT(1,:,:) = -(T(2,:,:) - T(1,:,:))   / dx;
gB(I,:,:) =  (T(I,:,:) - T(I-1,:,:)) / dx;

% k=1:   n = (0,0,-1) => n·∇T = -T_z ≈ -(T(:,:,2)-T(:,:,1))/dz
% k=K:   n = (0,0,+1) => n·∇T = +T_z ≈ +(T(:,:,K)-T(:,:,K-1))/dz
gF(:,:,1)  = -(T(:,:,2) - T(:,:,1))   / dz;
gBk(:,:,K) =  (T(:,:,K) - T(:,:,K-1)) / dz;

% Protect denominators from near-zero values
gL  = protect_denom(gL,  epsg);
gR  = protect_denom(gR,  epsg);
gT  = protect_denom(gT,  epsg);
gB  = protect_denom(gB,  epsg);
gF  = protect_denom(gF,  epsg);
gBk = protect_denom(gBk, epsg);

%% =========================================================
% Face values: beta = res / (n·∇T)
%% =========================================================
beta(:,1,:)   = res(:,1,:)   ./ gL(:,1,:);
beta(:,J,:)   = res(:,J,:)   ./ gR(:,J,:);
beta(1,:,:)   = res(1,:,:)   ./ gT(1,:,:);
beta(I,:,:)   = res(I,:,:)   ./ gB(I,:,:);
beta(:,:,1)   = res(:,:,1)   ./ gF(:,:,1);
beta(:,:,K)   = res(:,:,K)   ./ gBk(:,:,K);

%% =========================================================
% Edges: average of two adjacent faces
%% =========================================================

% (i,j)-edges for all k
beta(1,1,:) = 0.5*( res(1,1,:) ./ gL(1,1,:) + res(1,1,:) ./ gT(1,1,:) );
beta(I,1,:) = 0.5*( res(I,1,:) ./ gL(I,1,:) + res(I,1,:) ./ gB(I,1,:) );
beta(1,J,:) = 0.5*( res(1,J,:) ./ gR(1,J,:) + res(1,J,:) ./ gT(1,J,:) );
beta(I,J,:) = 0.5*( res(I,J,:) ./ gR(I,J,:) + res(I,J,:) ./ gB(I,J,:) );

% (i,k)-edges for all j
beta(1,:,1) = 0.5*( res(1,:,1) ./ gT(1,:,1) + res(1,:,1) ./ gF(1,:,1) );
beta(I,:,1) = 0.5*( res(I,:,1) ./ gB(I,:,1) + res(I,:,1) ./ gF(I,:,1) );
beta(1,:,K) = 0.5*( res(1,:,K) ./ gT(1,:,K) + res(1,:,K) ./ gBk(1,:,K) );
beta(I,:,K) = 0.5*( res(I,:,K) ./ gB(I,:,K) + res(I,:,K) ./ gBk(I,:,K) );

% (j,k)-edges for all i
beta(:,1,1) = 0.5*( res(:,1,1) ./ gL(:,1,1) + res(:,1,1) ./ gF(:,1,1) );
beta(:,J,1) = 0.5*( res(:,J,1) ./ gR(:,J,1) + res(:,J,1) ./ gF(:,J,1) );
beta(:,1,K) = 0.5*( res(:,1,K) ./ gL(:,1,K) + res(:,1,K) ./ gBk(:,1,K) );
beta(:,J,K) = 0.5*( res(:,J,K) ./ gR(:,J,K) + res(:,J,K) ./ gBk(:,J,K) );

%% =========================================================
% Corners: average of three faces
%% =========================================================
beta(1,1,1) = ( res(1,1,1)/gL(1,1,1) + res(1,1,1)/gT(1,1,1) + res(1,1,1)/gF(1,1,1) )/3;
beta(I,1,1) = ( res(I,1,1)/gL(I,1,1) + res(I,1,1)/gB(I,1,1) + res(I,1,1)/gF(I,1,1) )/3;
beta(1,J,1) = ( res(1,J,1)/gR(1,J,1) + res(1,J,1)/gT(1,J,1) + res(1,J,1)/gF(1,J,1) )/3;
beta(I,J,1) = ( res(I,J,1)/gR(I,J,1) + res(I,J,1)/gB(I,J,1) + res(I,J,1)/gF(I,J,1) )/3;

beta(1,1,K) = ( res(1,1,K)/gL(1,1,K) + res(1,1,K)/gT(1,1,K) + res(1,1,K)/gBk(1,1,K) )/3;
beta(I,1,K) = ( res(I,1,K)/gL(I,1,K) + res(I,1,K)/gB(I,1,K) + res(I,1,K)/gBk(I,1,K) )/3;
beta(1,J,K) = ( res(1,J,K)/gR(1,J,K) + res(1,J,K)/gT(1,J,K) + res(1,J,K)/gBk(1,J,K) )/3;
beta(I,J,K) = ( res(I,J,K)/gR(I,J,K) + res(I,J,K)/gB(I,J,K) + res(I,J,K)/gBk(I,J,K) )/3;

%% =========================================================
% Upwind coefficients for interior transport solve
% Discretize: T_x beta_x + T_y beta_y + T_z beta_z = 0
%% =========================================================
Tcoef = T;
if squareoption == 1
    Tcoef = Tcoef.^2;
end

% a_* for x-direction (i), b_* for y-direction (j), c_* for z-direction (k)
a_p = zeros(I,J,K); a_m = a_p;
b_p = zeros(I,J,K); b_m = b_p;
c_p = zeros(I,J,K); c_m = c_p;

for i = 2:I-1
    for j = 2:J-1
        for k = 2:K-1
            % one-sided derivatives of Tcoef
            Tx_p = -(Tcoef(i+1,j,k) - Tcoef(i,j,k)) / dx;
            Tx_m = -(Tcoef(i,j,k) - Tcoef(i-1,j,k)) / dx;
            Ty_p = -(Tcoef(i,j+1,k) - Tcoef(i,j,k)) / dy;
            Ty_m = -(Tcoef(i,j,k) - Tcoef(i,j-1,k)) / dy;
            Tz_p = -(Tcoef(i,j,k+1) - Tcoef(i,j,k)) / dz;
            Tz_m = -(Tcoef(i,j,k) - Tcoef(i,j,k-1)) / dz;

            % split into upwind positive/negative parts
            a_p(i,j,k) = max(Tx_p, 0);  a_m(i,j,k) = min(Tx_m, 0);
            b_p(i,j,k) = max(Ty_p, 0);  b_m(i,j,k) = min(Ty_m, 0);
            c_p(i,j,k) = max(Tz_p, 0);  c_m(i,j,k) = min(Tz_m, 0);
        end
    end
end

%% =========================================================
% Fast sweeping (8 sweep directions in 3D)
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

                denom = (a_p(i,j,k) - a_m(i,j,k))/dx ...
                      + (b_p(i,j,k) - b_m(i,j,k))/dy ...
                      + (c_p(i,j,k) - c_m(i,j,k))/dz;

                if abs(denom) < 1e-14
                    continue;
                end

                num = (a_p(i,j,k)*beta(i-1,j,k) - a_m(i,j,k)*beta(i+1,j,k))/dx ...
                    + (b_p(i,j,k)*beta(i,j-1,k) - b_m(i,j,k)*beta(i,j+1,k))/dy ...
                    + (c_p(i,j,k)*beta(i,j,k-1) - c_m(i,j,k)*beta(i,j,k+1))/dz;

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