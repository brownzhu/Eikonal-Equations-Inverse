function V = lambda_2d(T, T_star, dx, dy)
% T: traveltime, size nx x ny
% T_star: traveltime measured on the boundary
% dx, dy: grid spacing

[nx,ny] = size(T);
epsilon = 1e-6;
V = ones(nx, ny);

%% boundary conditions

% x-boundaries
for j = 1:ny
    V(1, j) = (T_star(1, j) - T(1, j)) ...
        / (-(T(2, j) - T(1, j)) / dx);
    V(nx, j) = (T_star(nx, j) - T(nx, j)) ...
        / ((T(nx, j) - T(nx-1, j)) / dx);
end

% y-boundaries
for i = 1:nx
    V(i, 1) = (T_star(i, 1) - T(i, 1)) ...
        / (-(T(i, 2) - T(i, 1)) / dy);
    V(i, ny) = (T_star(i, ny) - T(i, ny)) ...
        / ((T(i, ny) - T(i, ny-1)) / dy);
end

%% coefficients

T2 = T.^2;

Tx = 0*T;
Tx(2:end-1,2:end-1) = ...
    -(T2(3:end,2:end-1) - T2(1:end-2,2:end-1))/(2*dx);

Ty = 0*T;
Ty(2:end-1,2:end-1) = ...
    -(T2(2:end-1,3:end) - T2(2:end-1,1:end-2))/(2*dy);

%% iteration

iterv = 0;
num = 1.0;

while iterv < 300
    Vsave = V;

    % ---------- sweep 1: ( +x , +y )
    for i = 2:nx-1
        for j = 2:ny-1
            uxp = (V(i+1,j) - V(i,j)) / dx;
            uxm = (V(i,j) - V(i-1,j)) / dx;
            uyp = (V(i,j+1) - V(i,j)) / dy;
            uym = (V(i,j) - V(i,j-1)) / dy;

            ax1 = 1e-6 + num*abs(Tx(i,j));
            ay1 = 1e-6 + num*abs(Ty(i,j));

            V(i,j) = 1/(ax1/dx + ay1/dy) * ( ...
                -((uxp+uxm)/2 * Tx(i,j) + (uyp+uym)/2 * Ty(i,j)) ...
                + ax1/(2*dx) * (2*V(i,j) + (uxp-uxm)*dx) ...
                + ay1/(2*dy) * (2*V(i,j) + (uyp-uym)*dy) );
        end
    end

    % ---------- sweep 2: ( -x , -y )
    for i = nx-1:-1:2
        for j = ny-1:-1:2
            uxp = (V(i+1,j) - V(i,j)) / dx;
            uxm = (V(i,j) - V(i-1,j)) / dx;
            uyp = (V(i,j+1) - V(i,j)) / dy;
            uym = (V(i,j) - V(i,j-1)) / dy;

            ax1 = 1e-6 + num*abs(Tx(i,j));
            ay1 = 1e-6 + num*abs(Ty(i,j));

            V(i,j) = 1/(ax1/dx + ay1/dy) * ( ...
                -((uxp+uxm)/2 * Tx(i,j) + (uyp+uym)/2 * Ty(i,j)) ...
                + ax1/(2*dx) * (2*V(i,j) + (uxp-uxm)*dx) ...
                + ay1/(2*dy) * (2*V(i,j) + (uyp-uym)*dy) );
        end
    end

    % ---------- sweep 3: ( +x , -y )
    for i = 2:nx-1
        for j = ny-1:-1:2
            uxp = (V(i+1,j) - V(i,j)) / dx;
            uxm = (V(i,j) - V(i-1,j)) / dx;
            uyp = (V(i,j+1) - V(i,j)) / dy;
            uym = (V(i,j) - V(i,j-1)) / dy;

            ax1 = 1e-6 + num*abs(Tx(i,j));
            ay1 = 1e-6 + num*abs(Ty(i,j));

            V(i,j) = 1/(ax1/dx + ay1/dy) * ( ...
                -((uxp+uxm)/2 * Tx(i,j) + (uyp+uym)/2 * Ty(i,j)) ...
                + ax1/(2*dx) * (2*V(i,j) + (uxp-uxm)*dx) ...
                + ay1/(2*dy) * (2*V(i,j) + (uyp-uym)*dy) );
        end
    end

    % ---------- sweep 4: ( -x , +y )
    for i = nx-1:-1:2
        for j = 2:ny-1
            uxp = (V(i+1,j) - V(i,j)) / dx;
            uxm = (V(i,j) - V(i-1,j)) / dx;
            uyp = (V(i,j+1) - V(i,j)) / dy;
            uym = (V(i,j) - V(i,j-1)) / dy;

            ax1 = 1e-6 + num*abs(Tx(i,j));
            ay1 = 1e-6 + num*abs(Ty(i,j));

            V(i,j) = 1/(ax1/dx + ay1/dy) * ( ...
                -((uxp+uxm)/2 * Tx(i,j) + (uyp+uym)/2 * Ty(i,j)) ...
                + ax1/(2*dx) * (2*V(i,j) + (uxp-uxm)*dx) ...
                + ay1/(2*dy) * (2*V(i,j) + (uyp-uym)*dy) );
        end
    end

    err = max(max(abs(V - Vsave)));
    if err < 10*epsilon
        break
    end

    iterv = iterv + 1;
end
