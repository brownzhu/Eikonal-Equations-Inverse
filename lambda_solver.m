function lambda = lambda_solver(T, T_star, dx, dy, I, J)
% T: traveltime
% T_star: traveltime measured on the boundary
% dx,dy: grid size
% I,J: size

iterFS = 1000;
tolFS  = 1e-9;

% ====== debug switches ======
doCheck = false;        % 想静默就 false
epsg    = 1e-12;       % 分母保护阈值（按量纲可调，比如 1e-10）
% ============================

% Check matrix dimensions
[T_rows, T_cols] = size(T);
[Ts_rows, Ts_cols] = size(T_star);
I = I(1); J = J(1);  % Ensure I and J are scalars
if doCheck
    fprintf("[check] T size: %d x %d, expected: %d x %d\n", T_rows, T_cols, I, J);
    fprintf("[check] T_star size: %d x %d, expected: %d x %d\n", Ts_rows, Ts_cols, I, J);
end
if T_rows ~= I || T_cols ~= J
    error('lambda_solver: T size (%d x %d) does not match I x J (%d x %d)', T_rows, T_cols, I, J);
end
if Ts_rows ~= I || Ts_cols ~= J
    error('lambda_solver: T_star size (%d x %d) does not match I x J (%d x %d)', Ts_rows, Ts_cols, I, J);
end

if doCheck
    % ---- check T itself (zeros etc.) ----
    nZeroT = nnz(T==0);
    fprintf("[check] T: #zeros=%d, min(T)=%g, min_positive(T)=%g\n", ...
        nZeroT, min(T(:)), min(T(T>0)));
    nZeroTs = nnz(T_star==0);
    fprintf("[check] T_star: #zeros=%d, min(T_star)=%g, min_positive(T_star)=%g\n", ...
        nZeroTs, min(T_star(:)), min(T_star(T_star>0)));

    if any(~isfinite(T(:))) || any(~isfinite(T_star(:)))
        warning("[check] T or T_star has NaN/Inf!");
    end
end

%% ====== boundary condition with diagnostics & protection ======
lambda = ones(I, J);

% residuals on boundaries
rL = T_star(:,1) - T(:,1);
rR = T_star(:,J) - T(:,J);
rT = T_star(1,:) - T(1,:);
rB = T_star(I,:) - T(I,:);

% outward n·grad T (one-sided)
gL = -(T(:,2)   - T(:,1))   / dx;
gR =  (T(:,J)   - T(:,J-1)) / dx;
gT = -(T(2,:)   - T(1,:))   / dy;
gB =  (T(I,:)   - T(I-1,:)) / dy;

if doCheck
    fprintf("[check] min|gL|=%e, min|gR|=%e, min|gT|=%e, min|gB|=%e\n", ...
        min(abs(gL)), min(abs(gR)), min(abs(gT)), min(abs(gB)));
    fprintf("[check] #(gL==0)=%d #(gR==0)=%d #(gT==0)=%d #(gB==0)=%d\n", ...
        nnz(gL==0), nnz(gR==0), nnz(gT==0), nnz(gB==0));
    fprintf("[check] #(rL==0)=%d #(rR==0)=%d #(rT==0)=%d #(rB==0)=%d\n", ...
        nnz(rL==0), nnz(rR==0), nnz(rT==0), nnz(rB==0));
end

% protect denominators (avoid division blow-up)
gL = protect_denom(gL, epsg);
gR = protect_denom(gR, epsg);
gT = protect_denom(gT, epsg);
gB = protect_denom(gB, epsg);

% assign boundaries
lambda(:,1) = rL ./ gL;
lambda(:,J) = rR ./ gR;
lambda(1,:) = rT ./ gT;
lambda(I,:) = rB ./ gB;

% corners: average two directions to avoid overwrite instability
lambda(1,1) = 0.5*(rL(1)/gL(1) + rT(1)/gT(1));
lambda(I,1) = 0.5*(rL(I)/gL(I) + rB(1)/gB(1));
lambda(1,J) = 0.5*(rR(1)/gR(1) + rT(J)/gT(J));
lambda(I,J) = 0.5*(rR(I)/gR(I) + rB(J)/gB(J));

if doCheck
    fprintf("[check] boundary lambda: min=%e max=%e #NaN=%d #Inf=%d #zeros=%d\n", ...
        min(lambda(:)), max(lambda(:)), nnz(isnan(lambda(:))), nnz(isinf(lambda(:))), nnz(lambda(:)==0));
end

%% ====== preparation (your original) ======
a_plus  = zeros(I, J);
a_minus = zeros(I, J);
b_plus  = zeros(I, J);
b_minus = zeros(I, J);

for i = 2:I-1
    for j = 2:J-1
        a_plus(i, j)  = -(T(i+1, j) - T(i, j)) / dx;
        b_plus(i, j)  = -(T(i, j+1) - T(i, j)) / dy;
        a_minus(i, j) = -(T(i, j) - T(i-1, j)) / dx;
        b_minus(i, j) = -(T(i, j) - T(i, j-1)) / dy;
    end
end

a_plus_p  = (a_plus  + abs(a_plus))  / 2;
a_plus_m  = (a_plus  - abs(a_plus))  / 2;
a_minus_p = (a_minus + abs(a_minus)) / 2;
a_minus_m = (a_minus - abs(a_minus)) / 2;
b_plus_p  = (b_plus  + abs(b_plus))  / 2;
b_plus_m  = (b_plus  - abs(b_plus))  / 2;
b_minus_p = (b_minus + abs(b_minus)) / 2;
b_minus_m = (b_minus - abs(b_minus)) / 2;

%% ====== fast sweeping (your original + a bit of checking) ======
interOrder = {
    [2, 1, I-1, 2, 1, J-1],...
    [2, 1, I-1, J-1, -1, 2],...
    [I-1, -1, 2, 2, 1, J-1],...
    [I-1, -1, 2, J-1, -1, 2]
};

inter_erro = zeros(iterFS, 1);
for k = 1:iterFS

    lambda_temp = lambda;
    order = interOrder{cast(mod(k-1, 4), "uint8") + 1};

    for i = order(1):order(2):order(3)
        for j = order(4):order(5):order(6)

            temp1 = (a_plus_p(i, j) - a_minus_m(i, j)) / dx + (b_plus_p(i, j) - b_minus_m(i, j)) / dy;
            temp2 = (a_minus_p(i, j) * lambda(i-1, j) - a_plus_m(i, j) * lambda(i+1, j)) / dx;
            temp3 = (b_minus_p(i, j) * lambda(i, j-1) - b_plus_m(i, j) * lambda(i, j+1)) / dy;

            % protect temp1
            if abs(temp1) < eps
                temp1 = sign(temp1 + (temp1==0))*eps;
            end

            lambda(i, j) = (temp2 + temp3) / temp1;

        end
    end

    inter_erro(k) = norm(lambda - lambda_temp) * dx * dx;

    if doCheck && (mod(k,50)==0 || k==1)
        fprintf("[iter %d] err=%e, lambda: min=%e max=%e #NaN=%d #Inf=%d\n", ...
            k, inter_erro(k), min(lambda(:)), max(lambda(:)), nnz(isnan(lambda(:))), nnz(isinf(lambda(:))));
    end

    if inter_erro(k) < tolFS
        if doCheck
            fprintf("[done] converged at k=%d, err=%e\n", k, inter_erro(k));
        end
        break;
    end
end

if doCheck
    fprintf("[final] lambda: min=%e max=%e #NaN=%d #Inf=%d #zeros=%d\n", ...
        min(lambda(:)), max(lambda(:)), nnz(isnan(lambda(:))), nnz(isinf(lambda(:))), nnz(lambda(:)==0));
end

end


% ===== helper =====
function g = protect_denom(g, epsg)
% replace tiny denom by signed epsg (keep sign if possible)
mask = abs(g) < epsg;
if any(mask(:))
    g_masked = g(mask);
    s = sign(g_masked);
    s(s==0) = 1;
    g(mask) = s .* epsg;
end
end
% function lambda = lambda_solver(T, T_star, dx, dy, I, J)
% % T: traveltime
% % T_star: traveltime measured on the boundary
% % dx: length of division in the x direction
% % dy: length of division in the y direction
% % I: number of rows
% % J: number of columns
% 
% iterFS = 1000;
% tolFS = 1e-9;
% %% boundary condition
% lambda = ones(I, J);
% for i = 1:I
%     lambda(i, 1) = (T_star(i, 1) - T(i, 1)) / (-(T(i, 2) - T(i, 1)) / dx);  % left
%     lambda(i, J) = (T_star(i, J) - T(i, J)) / ((T(i, J) - T(i, J - 1)) / dx);   % right
% end
% for j = 1:J
%     lambda(1, j) = (T_star(1, j) - T(1, j)) / (-(T(2, j) - T(1, j)) / dy); % top
%     lambda(I, j) = (T_star(I, j) - T(I, j)) / ((T(I, j) - T(I - 1, j)) / dy); % bottom
% end
% 
% 
% %% preparation
% a_plus = zeros(I, J);
% a_minus = zeros(I, J);
% b_plus = zeros(I, J);
% b_minus = zeros(I, J);
% 
% for i = 2:I-1
%     for j = 2:J-1
%         a_plus(i, j) = -(T(i+1, j) - T(i, j)) / dx;
%         b_plus(i, j) = -(T(i, j+1) - T(i, j)) / dy;
%         a_minus(i, j) = -(T(i, j) - T(i-1, j)) / dx;
%         b_minus(i, j) = -(T(i, j) - T(i, j-1)) / dy;
%     end
% end
% 
% a_plus_p = (a_plus + abs(a_plus)) / 2;
% a_plus_m = (a_plus - abs(a_plus)) / 2;
% a_minus_p = (a_minus + abs(a_minus)) / 2;
% a_minus_m = (a_minus - abs(a_minus)) / 2;
% b_plus_p = (b_plus + abs(b_plus)) / 2;
% b_plus_m = (b_plus - abs(b_plus)) / 2;
% b_minus_p = (b_minus + abs(b_minus)) / 2;
% b_minus_m = (b_minus - abs(b_minus)) / 2;
% 
% 
% %% fast sweeping
% interOrder = {
%     [2, 1, I-1, 2, 1, J-1],...
%     [2, 1, I-1, J-1, -1, 2],...
%     [I-1, -1, 2, 2, 1, J-1],...
%     [I-1, -1, 2, J-1, -1, 2]
% };
% 
% inter_erro = zeros(iterFS, 1);
% for k = 1: iterFS
% 
%     lambda_temp = lambda;
%     order = interOrder{cast(mod(k-1, 4), "uint8") + 1};
%     for i = order(1): order(2): order(3)
%         for j = order(4): order(5): order(6)
%             temp1 = (a_plus_p(i, j) - a_minus_m(i, j)) / dx + (b_plus_p(i, j) - b_minus_m(i, j)) / dy;
%             temp2 = (a_minus_p(i, j) * lambda(i-1, j) - a_plus_m(i, j) * lambda(i+1, j) ) / dx;
%             temp3 = (b_minus_p(i, j) * lambda(i, j-1) - b_plus_m(i, j) * lambda(i, j+1) ) / dy;
%             if temp1==0
%                 temp1 = eps;
%             end
%             lambda(i, j) = temp1^(-1) * (temp2 + temp3);
% 
%         end
%     end
% 
%     inter_erro(k) = norm(lambda - lambda_temp)*dx*dx;
%     if inter_erro(k) < tolFS
%         break;
%     end
% 
% end
% 
% 
% end
