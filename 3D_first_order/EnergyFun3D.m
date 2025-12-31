function E = EnergyFun3D(T, T_star, dx, dy, dz)

[I, J, K] = size(T);

% x = const faces (i = 1, I)
E_x = sum(sum(sum( ...
        (T([1, I], 1:J-1, 1:K-1) - T_star([1, I], 1:J-1, 1:K-1)).^2 ...
      ))) * dy * dz;

% y = const faces (j = 1, J)
E_y = sum(sum(sum( ...
        (T(1:I-1, [1, J], 1:K-1) - T_star(1:I-1, [1, J], 1:K-1)).^2 ...
      ))) * dx * dz;

% z = const faces (k = 1, K)
E_z = sum(sum(sum( ...
        (T(1:I-1, 1:J-1, [1, K]) - T_star(1:I-1, 1:J-1, [1, K])).^2 ...
      ))) * dx * dy;

E = 0.5 * (E_x + E_y + E_z);

end
