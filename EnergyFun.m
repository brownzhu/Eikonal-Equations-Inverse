% EnergyFun
function E = EnergyFun(T, T_star, dx, dy)

[I, J] = size(T);
E1 = sum(sum((T(1: I-1, [1, J]) - T_star(1: I-1, [1, J])).^2)) * dy;
E2 = sum(sum((T([1, I], 1: J-1) - T_star([1, I], 1: J-1)).^2)) * dx;
E = (E1 + E2)*0.5;

end