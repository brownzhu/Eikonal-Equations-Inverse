function lambda = lambda_solver_3d_new(T, T_star, dx, dy, dz)
% T: traveltime, size I x J x K
% T_star: traveltime measured on the boundary
% dx, dy, dz: grid spacing
% I, J, K: grid size in x, y, z directions
[I,J,K] = size(T);
iterFS = 1000;
tolFS  = 1e-9;

%% boundary condition
lambda = ones(I, J, K);
if 1
% x-boundaries
for j = 1:J
    for k = 1:K
        lambda(1, j, k) = (T_star(1, j, k) - T(1, j, k)) ...
            / (-(T(2, j, k) - T(1, j, k)) / dx);
        lambda(I, j, k) = (T_star(I, j, k) - T(I, j, k)) ...
            / ((T(I, j, k) - T(I-1, j, k)) / dx);
    end
end

% y-boundaries
for i = 1:I
    for k = 1:K
        lambda(i, 1, k) = (T_star(i, 1, k) - T(i, 1, k)) ...
            / (-(T(i, 2, k) - T(i, 1, k)) / dy);
        lambda(i, J, k) = (T_star(i, J, k) - T(i, J, k)) ...
            / ((T(i, J, k) - T(i, J-1, k)) / dy);
    end
end

% z-boundaries
for i = 1:I
    for j = 1:J
        lambda(i, j, 1) = (T_star(i, j, 1) - T(i, j, 1)) ...
            / (-(T(i, j, 2) - T(i, j, 1)) / dz);
        lambda(i, j, K) = (T_star(i, j, K) - T(i, j, K)) ...
            / ((T(i, j, K) - T(i, j, K-1)) / dz);
    end
end
end
% if 1
% %     lambda=1./T.^2;
% %     lambda(2:end-1,2:end-1,2:end-1)=1+0*T(2:end-1,2:end-1,2:end-1);
% [x,y,z]=ndgrid(-1:dx:1,-1:dy:1,-1:dz:1);
% lambda=atan((z)./sqrt((x).^2+(y).^2)+eps);
% %   lambda(2:end-1,2:end-1,2:end-1)=20+0*T(2:end-1,2:end-1,2:end-1);
% 
% end

%% preparation
a_plus  = zeros(I, J, K);
a_minus = zeros(I, J, K);
b_plus  = zeros(I, J, K);
b_minus = zeros(I, J, K);
c_plus  = zeros(I, J, K);
c_minus = zeros(I, J, K);

for i = 2:I-1
    for j = 2:J-1
        for k = 2:K-1
            a_plus(i,j,k)  = -(T(i+1,j,k) - T(i,j,k)) / dx;
            a_minus(i,j,k) = -(T(i,j,k) - T(i-1,j,k)) / dx;
            b_plus(i,j,k)  = -(T(i,j+1,k) - T(i,j,k)) / dy;
            b_minus(i,j,k) = -(T(i,j,k) - T(i,j-1,k)) / dy;
            c_plus(i,j,k)  = -(T(i,j,k+1) - T(i,j,k)) / dz;
            c_minus(i,j,k) = -(T(i,j,k) - T(i,j,k-1)) / dz;
        end
    end
end

a_plus_p  = (a_plus  + abs(a_plus )) / 2;
a_plus_m  = (a_plus  - abs(a_plus )) / 2;
a_minus_p = (a_minus + abs(a_minus)) / 2;
a_minus_m = (a_minus - abs(a_minus)) / 2;

b_plus_p  = (b_plus  + abs(b_plus )) / 2;
b_plus_m  = (b_plus  - abs(b_plus )) / 2;
b_minus_p = (b_minus + abs(b_minus)) / 2;
b_minus_m = (b_minus - abs(b_minus)) / 2;

c_plus_p  = (c_plus  + abs(c_plus )) / 2;
c_plus_m  = (c_plus  - abs(c_plus )) / 2;
c_minus_p = (c_minus + abs(c_minus)) / 2;
c_minus_m = (c_minus - abs(c_minus)) / 2;


%% fast sweeping order (8 directions in 3D)
interOrder = {
    [2,1,I-1, 2,1,J-1, 2,1,K-1], ...
    [2,1,I-1, 2,1,J-1, K-1,-1,2], ...
    [2,1,I-1, J-1,-1,2, 2,1,K-1], ...
    [2,1,I-1, J-1,-1,2, K-1,-1,2], ...
    [I-1,-1,2, 2,1,J-1, 2,1,K-1], ...
    [I-1,-1,2, 2,1,J-1, K-1,-1,2], ...
    [I-1,-1,2, J-1,-1,2, 2,1,K-1], ...
    [I-1,-1,2, J-1,-1,2, K-1,-1,2]
};

inter_erro = zeros(iterFS,1);

for iter = 1:iterFS
    lambda_old = lambda;
    order = interOrder{mod(iter-1,8)+1};

    for i = order(1):order(2):order(3)
        for j = order(4):order(5):order(6)
            for k = order(7):order(8):order(9)

                temp1 = (a_plus_p(i,j,k) - a_minus_m(i,j,k))/dx ...
                      + (b_plus_p(i,j,k) - b_minus_m(i,j,k))/dy ...
                      + (c_plus_p(i,j,k) - c_minus_m(i,j,k))/dz;

                temp2 = ( a_minus_p(i,j,k)*lambda(i-1,j,k) ...
                        - a_plus_m(i,j,k)*lambda(i+1,j,k) ) / dx;

                temp3 = ( b_minus_p(i,j,k)*lambda(i,j-1,k) ...
                        - b_plus_m(i,j,k)*lambda(i,j+1,k) ) / dy;

                temp4 = ( c_minus_p(i,j,k)*lambda(i,j,k-1) ...
                        - c_plus_m(i,j,k)*lambda(i,j,k+1) ) / dz;

                if temp1 == 0
                    temp1 = eps;
                end

                lambda(i,j,k) = (temp2 + temp3 + temp4) / temp1;
            end
        end
    end

    inter_erro(iter) = norm(lambda(:) - lambda_old(:)) * dx * dy * dz;
    if inter_erro(iter) < tolFS
        break;
    end
end

end
