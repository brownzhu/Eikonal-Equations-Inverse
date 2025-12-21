function lambda = lambda_solver(T, T_star, dx, dy, I, J)
% T: traveltime
% T_star: traveltime measured on the boundary
% dx: length of division in the x direction
% dy: length of division in the y direction
% I: number of rows
% J: number of columns

iterFS = 1000;
tolFS = 1e-9;
%% boundary condition
lambda = ones(I, J);
for i = 1:I
    lambda(i, 1) = (T_star(i, 1) - T(i, 1)) / (-(T(i, 2) - T(i, 1)) / dx);  % left
    lambda(i, J) = (T_star(i, J) - T(i, J)) / ((T(i, J) - T(i, J - 1)) / dx);   % right
end
for j = 1:J
    lambda(1, j) = (T_star(1, j) - T(1, j)) / (-(T(2, j) - T(1, j)) / dy); % top
    lambda(I, j) = (T_star(I, j) - T(I, j)) / ((T(I, j) - T(I - 1, j)) / dy); % bottom
end


%% preparation
a_plus = zeros(I, J);
a_minus = zeros(I, J);
b_plus = zeros(I, J);
b_minus = zeros(I, J);

for i = 2:I-1
    for j = 2:J-1
        a_plus(i, j) = -(T(i+1, j) - T(i, j)) / dx;
        b_plus(i, j) = -(T(i, j+1) - T(i, j)) / dy;
        a_minus(i, j) = -(T(i, j) - T(i-1, j)) / dx;
        b_minus(i, j) = -(T(i, j) - T(i, j-1)) / dy;
    end
end

a_plus_p = (a_plus + abs(a_plus)) / 2;
a_plus_m = (a_plus - abs(a_plus)) / 2;
a_minus_p = (a_minus + abs(a_minus)) / 2;
a_minus_m = (a_minus - abs(a_minus)) / 2;
b_plus_p = (b_plus + abs(b_plus)) / 2;
b_plus_m = (b_plus - abs(b_plus)) / 2;
b_minus_p = (b_minus + abs(b_minus)) / 2;
b_minus_m = (b_minus - abs(b_minus)) / 2;


%% fast sweeping
interOrder = {
    [2, 1, I-1, 2, 1, J-1],...
    [2, 1, I-1, J-1, -1, 2],...
    [I-1, -1, 2, 2, 1, J-1],...
    [I-1, -1, 2, J-1, -1, 2]
};

inter_erro = zeros(iterFS, 1);
for k = 1: iterFS
    
    lambda_temp = lambda;
    order = interOrder{cast(mod(k-1, 4), "uint8") + 1};
    for i = order(1): order(2): order(3)
        for j = order(4): order(5): order(6)
            temp1 = (a_plus_p(i, j) - a_minus_m(i, j)) / dx + (b_plus_p(i, j) - b_minus_m(i, j)) / dy;
            temp2 = (a_minus_p(i, j) * lambda(i-1, j) - a_plus_m(i, j) * lambda(i+1, j) ) / dx;
            temp3 = (b_minus_p(i, j) * lambda(i, j-1) - b_plus_m(i, j) * lambda(i, j+1) ) / dy;
            if temp1==0
                temp1 = eps;
            end
            lambda(i, j) = temp1^(-1) * (temp2 + temp3);
            
        end
    end
    
    inter_erro(k) = norm(lambda - lambda_temp)*dx*dx;
    if inter_erro(k) < tolFS
        break;
    end

end


end
