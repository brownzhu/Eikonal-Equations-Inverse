function cstar = cStarSolver(T, T_star, dx, dy, I, J, c)

    % lambdaf = lambda_solver(T, T_star, dx, dy, I, J);
    % lambda1 = lambda_solver(T, T + 1, dx, dy, I, J);
    % % lambda1 has lost of entry with zero value, need regularization.
    % 
    % beta = lambdaf ./ lambda1;
    % % ------------------------------------------------------------
    % % Normalized adjoint weight:
    % %   beta = λ_f / λ_1
    % %
    % % This removes geometric amplification effects from λ_f,
    % % yielding a scale-stable, dimensionless adjoint weight
    % % that reflects relative data mismatch only.
    % %
    % % Numerically, this acts as an implicit adjoint
    % % preconditioner and improves robustness of the gradient.
    % % ------------------------------------------------------------
    % cstar = -beta ./ c.^3;
    % niu = 1; % 1e-4;
    % cstar = c_solver2(zeros(I, J), cstar, dx, dy, niu);

    beta_ = beta_solver(T, T_star, dx, dy);   % 直接解 beta，不再算 lambda1
    cstar = -beta_ ./ c.^3;

    niu = 1;
    cstar = c_solver2(zeros(I, J), cstar, dx, dy, niu);
 
end
