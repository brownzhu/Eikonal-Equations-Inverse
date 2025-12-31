function beta = beta_solver_3d_ndgrid(T, T_star, dx, dy, dz,squareoption)
% BETA_SOLVER_3D
% 3D extension of beta_solver (2D) using upwind + fast sweeping
%
% PDE:
%   T_x beta_x + T_y beta_y + T_z beta_z = 0
%
% BC:
%   (n · grad T) beta = T_star - T
%
% Index:
%   i -> y (dy), j -> x (dx), k -> z (dz)
%   i  x dx; j y dy; k z dz
[I, J, K] = size(T);
if ~isequal(size(T_star), [I, J, K])
    error('beta_solver_3d: size(T_star) must match size(T)');
end

iterFS = 500;
tolFS  = 1e-10;
epsg   = 1e-12;

beta = zeros(I, J, K);
r = T_star - T;

%% =========================================================
% Boundary normal derivatives (six faces)
%% =========================================================
gL  = zeros(I,J,K);
gR  = zeros(I,J,K);
gT  = zeros(I,J,K);
gB  = zeros(I,J,K);
gF  = zeros(I,J,K);
gBk = zeros(I,J,K);

% y-faces
gL(:,1,:) = -(T(:,2,:)   - T(:,1,:))   / dy;
gR(:,J,:) =  (T(:,J,:)   - T(:,J-1,:)) / dy;

% x-faces
gT(1,:,:) = -(T(2,:,:)   - T(1,:,:))   / dx;
gB(I,:,:) =  (T(I,:,:)   - T(I-1,:,:)) / dx;

% z-faces
gF(:,:,1) = -(T(:,:,2)   - T(:,:,1))   / dz;
gBk(:,:,K)=  (T(:,:,K)   - T(:,:,K-1)) / dz;


% protect denominators
gL  = protect_denom(gL,  epsg);
gR  = protect_denom(gR,  epsg);
gT  = protect_denom(gT,  epsg);
gB  = protect_denom(gB,  epsg);
gF  = protect_denom(gF,  epsg);
gBk = protect_denom(gBk, epsg);

%% =========================================================
% Face values
%% =========================================================

beta(:,1,:)   = r(:,1,:)   ./ gL(:,1,:);
beta(:,J,:)   = r(:,J,:)   ./ gR(:,J,:);
beta(1,:,:)   = r(1,:,:)   ./ gT(1,:,:);
beta(I,:,:)   = r(I,:,:)   ./ gB(I,:,:);
beta(:,:,1)   = r(:,:,1)   ./ gF(:,:,1);
beta(:,:,K)   = r(:,:,K)   ./ gBk(:,:,K);

%% =========================================================
% Edges: average of two faces
%% =========================================================

% (i,j)-edges
beta(1,1,:)   = 0.5*( r(1,1,:)   ./ gL(1,1,:)   + r(1,1,:)   ./ gT(1,1,:) );
beta(I,1,:)   = 0.5*( r(I,1,:)   ./ gL(I,1,:)   + r(I,1,:)   ./ gB(I,1,:) );
beta(1,J,:)   = 0.5*( r(1,J,:)   ./ gR(1,J,:)   + r(1,J,:)   ./ gT(1,J,:) );
beta(I,J,:)   = 0.5*( r(I,J,:)   ./ gR(I,J,:)   + r(I,J,:)   ./ gB(I,J,:) );

% (i,k)-edges
beta(1,:,1)   = 0.5*( r(1,:,1)   ./ gT(1,:,1)   + r(1,:,1)   ./ gF(1,:,1) );
beta(I,:,1)   = 0.5*( r(I,:,1)   ./ gB(I,:,1)   + r(I,:,1)   ./ gF(I,:,1) );
beta(1,:,K)   = 0.5*( r(1,:,K)   ./ gT(1,:,K)   + r(1,:,K)   ./ gBk(1,:,K) );
beta(I,:,K)   = 0.5*( r(I,:,K)   ./ gB(I,:,K)   + r(I,:,K)   ./ gBk(I,:,K) );

% (j,k)-edges
beta(:,1,1)   = 0.5*( r(:,1,1)   ./ gL(:,1,1)   + r(:,1,1)   ./ gF(:,1,1) );
beta(:,J,1)   = 0.5*( r(:,J,1)   ./ gR(:,J,1)   + r(:,J,1)   ./ gF(:,J,1) );
beta(:,1,K)   = 0.5*( r(:,1,K)   ./ gL(:,1,K)   + r(:,1,K)   ./ gBk(:,1,K) );
beta(:,J,K)   = 0.5*( r(:,J,K)   ./ gR(:,J,K)   + r(:,J,K)   ./ gBk(:,J,K) );

%% =========================================================
% Corners: average of three faces
%% =========================================================

beta(1,1,1) = ( r(1,1,1)/gL(1,1,1) + r(1,1,1)/gT(1,1,1) + r(1,1,1)/gF(1,1,1) )/3;
beta(I,1,1) = ( r(I,1,1)/gL(I,1,1) + r(I,1,1)/gB(I,1,1) + r(I,1,1)/gF(I,1,1) )/3;
beta(1,J,1) = ( r(1,J,1)/gR(1,J,1) + r(1,J,1)/gT(1,J,1) + r(1,J,1)/gF(1,J,1) )/3;
beta(I,J,1) = ( r(I,J,1)/gR(I,J,1) + r(I,J,1)/gB(I,J,1) + r(I,J,1)/gF(I,J,1) )/3;

beta(1,1,K) = ( r(1,1,K)/gL(1,1,K) + r(1,1,K)/gT(1,1,K) + r(1,1,K)/gBk(1,1,K) )/3;
beta(I,1,K) = ( r(I,1,K)/gL(I,1,K) + r(I,1,K)/gB(I,1,K) + r(I,1,K)/gBk(I,1,K) )/3;
beta(1,J,K) = ( r(1,J,K)/gR(1,J,K) + r(1,J,K)/gT(1,J,K) + r(1,J,K)/gBk(1,J,K) )/3;
beta(I,J,K) = ( r(I,J,K)/gR(I,J,K) + r(I,J,K)/gB(I,J,K) + r(I,J,K)/gBk(I,J,K) )/3;

%% =========================================================
% Upwind coefficients
%% =========================================================
if squareoption==1
T=T.^2;
else
end
a_p = zeros(I,J,K); a_m = a_p;   % x
b_p = zeros(I,J,K); b_m = b_p;   % y
c_p = zeros(I,J,K); c_m = c_p;   % z

for i = 2:I-1
    for j = 2:J-1
        for k = 2:K-1
            ax_p = -(T(i+1,j,k)-T(i,j,k))/dx;
            ax_m = -(T(i,j,k)-T(i-1,j,k))/dx;
            ay_p = -(T(i,j+1,k)-T(i,j,k))/dy;
            ay_m = -(T(i,j,k)-T(i,j-1,k))/dy;
            az_p = -(T(i,j,k+1)-T(i,j,k))/dz;
            az_m = -(T(i,j,k)-T(i,j,k-1))/dz;

            a_p(i,j,k) = max(ax_p,0); a_m(i,j,k) = min(ax_m,0);
            b_p(i,j,k) = max(ay_p,0); b_m(i,j,k) = min(ay_m,0);
            c_p(i,j,k) = max(az_p,0); c_m(i,j,k) = min(az_m,0);
        end
    end
end

%% =========================================================
% Fast sweeping (8 directions)
%% =========================================================

orders = {
    [ 2  1 I-1  2  1 J-1  2  1 K-1 ]
    [ 2  1 I-1  2  1 J-1  K-1 -1 2 ]
    [ 2  1 I-1  J-1 -1 2  2  1 K-1 ]
    [ I-1 -1 2  2  1 J-1  2  1 K-1 ]
    [ I-1 -1 2  J-1 -1 2  K-1 -1 2 ]
    [ 2  1 I-1  J-1 -1 2  K-1 -1 2 ]
    [ I-1 -1 2  2  1 J-1  K-1 -1 2 ]
    [ I-1 -1 2  J-1 -1 2  2  1 K-1 ]
};

for it = 1:iterFS
    beta_old = beta;
    ord = orders{mod(it-1,8)+1};

    for i = ord(1):ord(2):ord(3)
        for j = ord(4):ord(5):ord(6)
            for k = ord(7):ord(8):ord(9)

                denom = (a_p(i,j,k)-a_m(i,j,k))/dx ...
                      + (b_p(i,j,k)-b_m(i,j,k))/dy ...
                      + (c_p(i,j,k)-c_m(i,j,k))/dz;

                if abs(denom) < 1e-14
                    continue
                end

                num = (a_p(i,j,k)*beta(i-1,j,k) - a_m(i,j,k)*beta(i+1,j,k))/dx ...
                    + (b_p(i,j,k)*beta(i,j-1,k) - b_m(i,j,k)*beta(i,j+1,k))/dy ...
                    + (c_p(i,j,k)*beta(i,j,k-1) - c_m(i,j,k)*beta(i,j,k+1))/dz;

                beta(i,j,k) = num / denom;
            end
        end
    end

    err = norm(beta(:)-beta_old(:)) * dx * dy * dz;
    if err < tolFS
        break
    end
end

end

%% =========================================================
function g = protect_denom(g, epsg)
mask = abs(g) < epsg;
if any(mask(:))
    s = sign(g(mask));
    s(s==0) = 1;
    g(mask) = s .* epsg;
end
end
