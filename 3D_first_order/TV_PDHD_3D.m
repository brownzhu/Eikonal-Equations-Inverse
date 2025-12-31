function [u,w1,w2,w3,Energy,Dgap,TimeCost,itr] = TV_PDHD_3D(w1,w2,w3,f,lbd,NIT,GapTol)
%TV_PDHD_3D  PDHD algorithm for 3D ROF denoising (dual projected gradient).
%
% This code solves a 3D Rudin–Osher–Fatemi (ROF) total-variation denoising model.
%
% -------------------- ROF primal problem --------------------
% We recover a denoised volume u from noisy data f by:
%
%   (P)   min_u  \int_\Omega |∇u| dx  +  (λ/2) * \int_\Omega (u - f)^2 dx
%
% where
%   - |∇u| is the isotropic TV seminorm (in 3D: sqrt(ux^2 + uy^2 + uz^2))
%   - λ > 0 balances smoothing (TV) and data fidelity (L2 fit to f)
%
% In THIS implementation, the input parameter "lbd" is exactly λ in (P),
% i.e. it multiplies the quadratic data-misfit term (u-f)^2.
%
% -------------------- Dual problem used here --------------------
% A standard dual form of ROF is:
%
%   (D)   max_{|w(x)|<=1}  <f, div w>  - (1/(2λ)) * ||div w||_2^2
%         (equivalently, adding constants: (λ/2)||f||^2 - (1/(2λ))||div w - λ f||^2)
%
% where
%   - w(x) is a vector field (dual variable) with pointwise constraint |w|<=1
%   - div is the divergence operator (negative adjoint of gradient under suitable BC)
%
% Relationship between primal and dual (KKT):
%   u = f - (1/λ) * div w.
%
% That is exactly the primal update target you see in the iteration:
%     u ≈ f - (1/lbd)*DivW
%
% -------------------- Algorithm idea (PDHD flavor) --------------------
% Each iteration does:
%   1) Dual gradient ascent/descent step on w using current grad(u)
%   2) Projection of w back onto the constraint set {|w|<=1} pointwise
%   3) Recompute div(w)
%   4) Update primal u by relaxation towards f - (1/λ)div(w)
%   5) Compute primal energy, dual energy, and a relative duality gap stopping test
%
% -------------------- Discretization (finite differences) --------------------
% We use forward differences for grad(u) with zero padding at the boundary.
% The divergence is implemented in the same "difference-combination" style
% as your original 2D code, to remain consistent.
%
% NOTE:
%   This is a practical discretization. For strict "discrete adjoint pairing"
%   one would enforce div = -grad^* exactly under a specific boundary condition.
%
% Inputs:
%   w1,w2,w3 : initial dual field components (same size as f)
%   f        : noisy 3D volume (m-by-n-by-p)
%   lbd      : λ fidelity weight in ROF (P)
%   NIT      : max iterations
%   GapTol   : stopping tolerance for relative duality gap
%
% Outputs:
%   u        : denoised volume (primal solution)
%   w1,w2,w3 : dual solution components
%   Energy   : final dual objective value
%   Dgap     : final relative duality gap
%   TimeCost : CPU time
%   itr      : number of iterations used

verbose = 0;
t0 = cputime;

[m,n,p] = size(f);

% -------------------- primal init --------------------
% Initialize u with the noisy data. (Common default for ROF.)
u = f;

% -------------------- grad(u): forward differences --------------------
% ux(i,j,k) = u(i,j+1,k) - u(i,j,k), and ux(:,end,:) = 0 (zero padding)
% uy(i,j,k) = u(i+1,j,k) - u(i,j,k), and uy(end,:,:) = 0
% uz(i,j,k) = u(i,j,k+1) - u(i,j,k), and uz(:,:,end) = 0
ux = cat(2, u(:,2:end,:) - u(:,1:end-1,:), zeros(m,1,p));
uy = cat(1, u(2:end,:,:) - u(1:end-1,:,:), zeros(1,n,p));
uz = cat(3, u(:,:,2:end) - u(:,:,1:end-1), zeros(m,n,1));

% -------------------- div(w): discrete divergence --------------------
% This matches your 2D pattern:
%   DivW = Dx^- w1 + Dy^- w2 + Dz^- w3
% implemented via differences with a boundary "injection" term:
%   along x: [w1(:,1,:), w1(:,2:end,:)-w1(:,1:end-1,:)]
% similarly for y and z.
DivW = cat(2, w1(:,1,:), w1(:,2:end,:) - w1(:,1:end-1,:)) ...
     + cat(1, w2(1,:,:), w2(2:end,:,:) - w2(1:end-1,:,:)) ...
     + cat(3, w3(:,:,1), w3(:,:,2:end) - w3(:,:,1:end-1));

% -------------------- dual objective --------------------
% Dual energy (up to an additive constant) in the form used by your 2D code:
%   Dual = (λ/2)*||f||^2 - (1/(2λ))*||DivW - λ f||^2
Dual   = (lbd/2) * (sum(f(:).^2) - (1/lbd^2)*sum((DivW(:) - lbd*f(:)).^2));

% -------------------- primal objective --------------------
% TV(u) + (λ/2)||u-f||^2
gu_norm = sqrt(ux.^2 + uy.^2 + uz.^2);
Primal = sum(gu_norm(:) + (lbd/2)*(u(:)-f(:)).^2);

% -------------------- relative duality gap --------------------
% A common stopping metric: (P-D)/( |P|+|D| ).
Dgap   = (Primal - Dual) / (abs(Primal) + abs(Dual));

for itr = 1:NIT

    % -------------------- choose step size tau --------------------
    % Your original code uses a heuristic schedule that increases with itr.
    % In 3D this can still work but is more sensitive (may need smaller bounds).
    tau = 0.2 + 0.08*itr;

    % -------------------- dual gradient step --------------------
    % w <- w - tau * λ * ∇u
    % (Sign convention depends on the exact dual form; we keep your 2D style.)
    w1 = w1 - tau*lbd*ux;
    w2 = w2 - tau*lbd*uy;
    w3 = w3 - tau*lbd*uz;

    % -------------------- projection onto |w|<=1 --------------------
    % Enforce pointwise constraint:
    %   w(i,j,k) := w(i,j,k) / max(1,|w(i,j,k)|)
    wnorm = max(1, sqrt(w1.^2 + w2.^2 + w3.^2));
    w1 = w1 ./ wnorm;
    w2 = w2 ./ wnorm;
    w3 = w3 ./ wnorm;

    % -------------------- update div(w) and dual energy --------------------
    DivW = cat(2, w1(:,1,:), w1(:,2:end,:) - w1(:,1:end-1,:)) ...
         + cat(1, w2(1,:,:), w2(2:end,:,:) - w2(1:end-1,:,:)) ...
         + cat(3, w3(:,:,1), w3(:,:,2:end) - w3(:,:,1:end-1));

    Dual = (lbd/2) * (sum(f(:).^2) - (1/lbd^2)*sum((DivW(:) - lbd*f(:)).^2));

    % -------------------- choose relaxation parameter theta --------------------
    % In your 2D code: theta depends on itr and tau.
    % It controls how aggressively u is moved towards the "dual-implied" primal:
    %   u_target = f - (1/λ)div(w)
    theta = (0.5 - 5.0/(15.0+itr)) / tau;

    % -------------------- primal update --------------------
    % u <- (1-theta)u + theta*( f - (1/λ)div(w) )
    % This is a relaxed fixed-point update consistent with KKT relation.
    u = (1.0-theta)*u + theta*(f - (1/lbd)*DivW);

    % -------------------- update grad(u), primal energy, gap --------------------
    ux = cat(2, u(:,2:end,:) - u(:,1:end-1,:), zeros(m,1,p));
    uy = cat(1, u(2:end,:,:) - u(1:end-1,:,:), zeros(1,n,p));
    uz = cat(3, u(:,:,2:end) - u(:,:,1:end-1), zeros(m,n,1));

    gu_norm = sqrt(ux.^2 + uy.^2 + uz.^2);
    Primal = sum(gu_norm(:) + (lbd/2)*(u(:)-f(:)).^2);

    Dgap = (Primal - Dual) / (abs(Primal) + abs(Dual));

    if verbose && (mod(itr,10)==0 || itr==1)
        fprintf('itr %4d: Pri=%8.3e, Dua=%8.3e, Gap=%5.2e\n', itr, Primal, Dual, Dgap);
    end

    if Dgap < GapTol
        break
    end
end

Energy = Dual;
TimeCost = cputime - t0;

end