function L2 = L2NormBoundary(T, T_star, dx, dy)
% Computes ||T - T_star||_{L2(∂Ω)}

[I, J] = size(T);

E_lr = sum( (T(1:I-1,1) - T_star(1:I-1,1)).^2 ...
          + (T(1:I-1,J) - T_star(1:I-1,J)).^2 ) * dy;

E_bt = sum( (T(1,1:J-1) - T_star(1,1:J-1)).^2 ...
          + (T(I,1:J-1) - T_star(I,1:J-1)).^2 ) * dx;

L2 = sqrt(E_lr + E_bt);

end