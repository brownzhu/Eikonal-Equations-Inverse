function T = TravelTime_solver(c, fixed_pt_list, dx, ~, I, J)

T = ones(I, J)*999;
for idx = 1:size(fixed_pt_list, 1)
    val = fixed_pt_list(idx, 1);
    j = fixed_pt_list(idx, 2); 
    i = fixed_pt_list(idx, 3);
    T(i, j) = val;
end


interOrder = { [1, 1, I, 1, 1, J],...
    [1, 1, I, J, -1, 1],...
    [I, -1, 1, 1, 1, J],...
    [I, -1, 1, J, -1, 1]
};

interFS = 1000;
tolFS = 1e-6;
inter_erro = zeros(interFS, 1);
for k = 1: interFS
    
    T_temp = T;
    order = interOrder{cast(mod(k-1, 4), "uint8") + 1};
    for i = order(1): order(2): order(3)
        for j = order(4): order(5): order(6)     
            
            
            if i ~= 1 && i ~= I
                a = min(T(i-1, j), T(i+1, j));
            else
                if i == 1
                    a = 2*T(i+1, j) - T(i+2, j);
                else
                    a = 2*T(i-1, j) - T(i-2, j);
                end
            end
            
            if j ~= 1 && j ~= J
                b = min(T(i, j-1), T(i, j+1));
            else
                if j == 1
                    b = 2*T(i, j+1) - T(i, j+2);
                else
                    b = 2*T(i, j-1) - T(i, j-2);
                end
            end
                
            
            if abs(a-b) < dx/c(i, j)
                Tij = (a + b+ sqrt(2*dx^2/(c(i, j))^2 - (a-b)^2)) / 2;
            else
                Tij = min(a, b) + dx/c(i, j);
            end
            T(i, j) = min(T(i, j), Tij);
            
        end
    end
    for idx = 1:size(fixed_pt_list, 1)
        val = fixed_pt_list(idx, 1);
        j = fixed_pt_list(idx, 2); 
        i = fixed_pt_list(idx, 3);
        T(i, j) = val;
    end
    
    inter_erro(k) = norm(T - T_temp)*dx*dx;
    if inter_erro(k) < tolFS
        break;
    end

end