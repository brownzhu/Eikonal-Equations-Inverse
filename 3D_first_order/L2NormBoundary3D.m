function L2 = L2NormBoundary3D(T, T_star, dx, dy,dz)
% Computes ||T - T_star||_{L2(∂Ω)}

[I, J,K] = size(T);

E_l = sum(sum( (T(1:I-1,1:J-1,1) - T_star(1:I-1,1:J-1,1)).^2 ...
          + (T(1:I-1,1:J-1,K) - T_star(1:I-1,1:J-1,K)).^2 )) * dx*dy;

E_2 = sum(sum( (T(1:I-1,1,1:K-1) - T_star(1:I-1,1,1:K-1)).^2 ...
          + (T(1:I-1,J,1:K-1) - T_star(1:I-1,J,1:K-1)).^2 )) * dx*dz;
    
E_3 = sum(sum( (T(1,1:J-1,1:K-1) - T_star(1,1:J-1,1:K-1)).^2 ...
          + (T(I,1:J-1,1:K-1) - T_star(I,1:J-1,1:K-1)).^2 )) * dy*dz;      
 
L2 = sqrt(E_l + E_2+E_3);

end