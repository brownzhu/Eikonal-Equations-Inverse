function V = lambda_3d(T, T_star, dx, dy, dz)
% T: traveltime, size I x J x K
% T_star: traveltime measured on the boundary
% dx, dy, dz: grid spacing

[nx,ny,nz] = size(T);
iterFS = 1000;
epsilon = 1e-6;
V = ones(nx, ny, nz);
if 1
% x-boundaries
for j = 1:ny
    for k = 1:nz
        V(1, j, k) = (T_star(1, j, k) - T(1, j, k)) ...
            / (-(T(2, j, k) - T(1, j, k)) / dx);
        V(nx, j, k) = (T_star(nx, j, k) - T(nx, j, k)) ...
            / ((T(nx, j, k) - T(nx-1, j, k)) / dx);
    end
end

% y-boundaries
for i = 1:nx
    for k = 1:nz
        V(i, 1, k) = (T_star(i, 1, k) - T(i, 1, k)) ...
            / (-(T(i, 2, k) - T(i, 1, k)) / dy);
        V(i, ny, k) = (T_star(i, ny, k) - T(i, ny, k)) ...
            / ((T(i, ny, k) - T(i, ny-1, k)) / dy);
    end
end

% z-boundaries
for i = 1:nx
    for j = 1:ny
        V(i, j, 1) = (T_star(i, j, 1) - T(i, j, 1)) ...
            / (-(T(i, j, 2) - T(i, j, 1)) / dz);
        V(i, j, nz) = (T_star(i, j, nz) - T(i, j, nz)) ...
            / ((T(i, j, nz) - T(i, j, nz-1)) / dz);
    end
end
end
%%

T2=T.^2;
Tx=0*T;
Tx(2:end-1,2:end-1,2:end-1)=-(T2(3:end,2:end-1,2:end-1)-T2(1:end-2,2:end-1,2:end-1))/2/dx;
Ty=0*T;
Ty(2:end-1,2:end-1,2:end-1)=-(T2(2:end-1,3:end,2:end-1)-T2(2:end-1,1:end-2,2:end-1))/2/dy;
Tz=0*T;
Tz(2:end-1,2:end-1,2:end-1)=-(T2(2:end-1,2:end-1,3:end)-T2(2:end-1,2:end-1,1:end-2))/2/dz;
iterv=0;
num=1.0;
while iterv <300
    Vsave=V;
    iterv;
    for i =2:nx-1
        for j = 2 : ny-1
            for k = 2: nz -1
                uxp=(V(i+1,j,k)-V(i,j,k))/dx;
                uxm=(V(i,j,k)-V(i-1,j,k))/dx;
                uyp=(V(i,j+1,k)-V(i,j,k))/dy;
                uym=(V(i,j,k)-V(i,j-1,k))/dy;
                uzp=(V(i,j,k+1)-V(i,j,k))/dz;
                uzm=(V(i,j,k)-V(i,j,k-1))/dz;
                ax1=1e-6+num*(abs(Tx(i,j,k)));
                ay1=1e-6+num*(abs(Ty(i,j,k)));
                az1=1e-6+num*(abs(Tz(i,j,k)));
                unew=1/(ax1/dx+ay1/dy+az1/dz)*(0-((uxp+uxm)/2*Tx(i,j,k)+(uyp+uym)/2*Ty(i,j,k)+(uzp+uzm)/2*Tz(i,j,k))+ax1/2/dx*(2*V(i,j,k)+(uxp-uxm)*dx)+ay1/2/dy*(2*V(i,j,k)...
                     +(uyp-uym)*dy)+az1/2/dz*(2*V(i,j,k)+(uzp-uzm)*dz));
                     V(i,j,k)=unew;
            end
        end
    end

 
    for i =nx-1:-1:2
        for j =  ny-1:-1:2
            for k = nz -1:-1:2
                uxp=(V(i+1,j,k)-V(i,j,k))/dx;
                uxm=(V(i,j,k)-V(i-1,j,k))/dx;
                uyp=(V(i,j+1,k)-V(i,j,k))/dy;
                uym=(V(i,j,k)-V(i,j-1,k))/dy;
                uzp=(V(i,j,k+1)-V(i,j,k))/dz;
                uzm=(V(i,j,k)-V(i,j,k-1))/dz;
                ax1=1e-6+num*(abs(Tx(i,j,k)));
                ay1=1e-6+num*(abs(Ty(i,j,k)));
                az1=1e-6+num*(abs(Tz(i,j,k)));
                unew=1/(ax1/dx+ay1/dy+az1/dz)*(0-((uxp+uxm)/2*Tx(i,j,k)+(uyp+uym)/2*Ty(i,j,k)+(uzp+uzm)/2*Tz(i,j,k))+ax1/2/dx*(2*V(i,j,k)+(uxp-uxm)*dx)+ay1/2/dy*(2*V(i,j,k)...
                     +(uyp-uym)*dy)+az1/2/dz*(2*V(i,j,k)+(uzp-uzm)*dz));
                     V(i,j,k)=unew;
            end
        end
    end

    
    
    for i =2:nx-1
        for j =   ny-1:-1:2
            for k = 2: nz -1
                              uxp=(V(i+1,j,k)-V(i,j,k))/dx;
                uxm=(V(i,j,k)-V(i-1,j,k))/dx;
                uyp=(V(i,j+1,k)-V(i,j,k))/dy;
                uym=(V(i,j,k)-V(i,j-1,k))/dy;
                uzp=(V(i,j,k+1)-V(i,j,k))/dz;
                uzm=(V(i,j,k)-V(i,j,k-1))/dz;
                ax1=1e-6+num*(abs(Tx(i,j,k)));
                ay1=1e-6+num*(abs(Ty(i,j,k)));
                az1=1e-6+num*(abs(Tz(i,j,k)));
                unew=1/(ax1/dx+ay1/dy+az1/dz)*(0-((uxp+uxm)/2*Tx(i,j,k)+(uyp+uym)/2*Ty(i,j,k)+(uzp+uzm)/2*Tz(i,j,k))+ax1/2/dx*(2*V(i,j,k)+(uxp-uxm)*dx)+ay1/2/dy*(2*V(i,j,k)...
                     +(uyp-uym)*dy)+az1/2/dz*(2*V(i,j,k)+(uzp-uzm)*dz));
                     V(i,j,k)=unew;
            end
        end
    end


    for i =nx-1:-1:2
        for j = 2: ny-1
            for k =  nz -1:-1:2
                            uxp=(V(i+1,j,k)-V(i,j,k))/dx;
                uxm=(V(i,j,k)-V(i-1,j,k))/dx;
                uyp=(V(i,j+1,k)-V(i,j,k))/dy;
                uym=(V(i,j,k)-V(i,j-1,k))/dy;
                uzp=(V(i,j,k+1)-V(i,j,k))/dz;
                uzm=(V(i,j,k)-V(i,j,k-1))/dz;
                ax1=1e-6+num*(abs(Tx(i,j,k)));
                ay1=1e-6+num*(abs(Ty(i,j,k)));
                az1=1e-6+num*(abs(Tz(i,j,k)));
                unew=1/(ax1/dx+ay1/dy+az1/dz)*(0-((uxp+uxm)/2*Tx(i,j,k)+(uyp+uym)/2*Ty(i,j,k)+(uzp+uzm)/2*Tz(i,j,k))+ax1/2/dx*(2*V(i,j,k)+(uxp-uxm)*dx)+ay1/2/dy*(2*V(i,j,k)...
                     +(uyp-uym)*dy)+az1/2/dz*(2*V(i,j,k)+(uzp-uzm)*dz));
                     V(i,j,k)=unew;
            end
        end
    end

    
    for i =2:nx-1
        for j =  ny-1:-1:2
            for k = nz -1:-1:2
                             uxp=(V(i+1,j,k)-V(i,j,k))/dx;
                uxm=(V(i,j,k)-V(i-1,j,k))/dx;
                uyp=(V(i,j+1,k)-V(i,j,k))/dy;
                uym=(V(i,j,k)-V(i,j-1,k))/dy;
                uzp=(V(i,j,k+1)-V(i,j,k))/dz;
                uzm=(V(i,j,k)-V(i,j,k-1))/dz;
                ax1=1e-6+num*(abs(Tx(i,j,k)));
                ay1=1e-6+num*(abs(Ty(i,j,k)));
                az1=1e-6+num*(abs(Tz(i,j,k)));
                unew=1/(ax1/dx+ay1/dy+az1/dz)*(0-((uxp+uxm)/2*Tx(i,j,k)+(uyp+uym)/2*Ty(i,j,k)+(uzp+uzm)/2*Tz(i,j,k))+ax1/2/dx*(2*V(i,j,k)+(uxp-uxm)*dx)+ay1/2/dy*(2*V(i,j,k)...
                     +(uyp-uym)*dy)+az1/2/dz*(2*V(i,j,k)+(uzp-uzm)*dz));
                     V(i,j,k)=unew;
            end
        end
    end


    for i =nx-1:-1:2
        for j = 2 : ny-1
            for k = 2: nz -1
                              uxp=(V(i+1,j,k)-V(i,j,k))/dx;
                uxm=(V(i,j,k)-V(i-1,j,k))/dx;
                uyp=(V(i,j+1,k)-V(i,j,k))/dy;
                uym=(V(i,j,k)-V(i,j-1,k))/dy;
                uzp=(V(i,j,k+1)-V(i,j,k))/dz;
                uzm=(V(i,j,k)-V(i,j,k-1))/dz;
                ax1=1e-6+num*(abs(Tx(i,j,k)));
                ay1=1e-6+num*(abs(Ty(i,j,k)));
                az1=1e-6+num*(abs(Tz(i,j,k)));
                unew=1/(ax1/dx+ay1/dy+az1/dz)*(0-((uxp+uxm)/2*Tx(i,j,k)+(uyp+uym)/2*Ty(i,j,k)+(uzp+uzm)/2*Tz(i,j,k))+ax1/2/dx*(2*V(i,j,k)+(uxp-uxm)*dx)+ay1/2/dy*(2*V(i,j,k)...
                     +(uyp-uym)*dy)+az1/2/dz*(2*V(i,j,k)+(uzp-uzm)*dz));
                     V(i,j,k)=unew;
            end
        end
    end

  
    
    for i =2:nx-1
        for j = 2 : ny-1
            for k =  nz -1:-1:2
                               uxp=(V(i+1,j,k)-V(i,j,k))/dx;
                uxm=(V(i,j,k)-V(i-1,j,k))/dx;
                uyp=(V(i,j+1,k)-V(i,j,k))/dy;
                uym=(V(i,j,k)-V(i,j-1,k))/dy;
                uzp=(V(i,j,k+1)-V(i,j,k))/dz;
                uzm=(V(i,j,k)-V(i,j,k-1))/dz;
                ax1=1e-6+num*(abs(Tx(i,j,k)));
                ay1=1e-6+num*(abs(Ty(i,j,k)));
                az1=1e-6+num*(abs(Tz(i,j,k)));
                unew=1/(ax1/dx+ay1/dy+az1/dz)*(0-((uxp+uxm)/2*Tx(i,j,k)+(uyp+uym)/2*Ty(i,j,k)+(uzp+uzm)/2*Tz(i,j,k))+ax1/2/dx*(2*V(i,j,k)+(uxp-uxm)*dx)+ay1/2/dy*(2*V(i,j,k)...
                     +(uyp-uym)*dy)+az1/2/dz*(2*V(i,j,k)+(uzp-uzm)*dz));
                     V(i,j,k)=unew;
            end
        end
    end

    
  
    for i =nx-1:-1:2
        for j = ny-1:-1:2
            for k = 2: nz -1
                               uxp=(V(i+1,j,k)-V(i,j,k))/dx;
                uxm=(V(i,j,k)-V(i-1,j,k))/dx;
                uyp=(V(i,j+1,k)-V(i,j,k))/dy;
                uym=(V(i,j,k)-V(i,j-1,k))/dy;
                uzp=(V(i,j,k+1)-V(i,j,k))/dz;
                uzm=(V(i,j,k)-V(i,j,k-1))/dz;
                ax1=1e-6+num*(abs(Tx(i,j,k)));
                ay1=1e-6+num*(abs(Ty(i,j,k)));
                az1=1e-6+num*(abs(Tz(i,j,k)));
                unew=1/(ax1/dx+ay1/dy+az1/dz)*(0-((uxp+uxm)/2*Tx(i,j,k)+(uyp+uym)/2*Ty(i,j,k)+(uzp+uzm)/2*Tz(i,j,k))+ax1/2/dx*(2*V(i,j,k)+(uxp-uxm)*dx)+ay1/2/dy*(2*V(i,j,k)...
                     +(uyp-uym)*dy)+az1/2/dz*(2*V(i,j,k)+(uzp-uzm)*dz));
                     V(i,j,k)=unew;
            end
        end
    end

    err=max(max(max(abs(V-Vsave))));
    if max(max(max(abs(V-Vsave))))<10*epsilon
        break
    else
        Vsave=V;
    end
    iterv=iterv+1;
   if iterv>150
   fprintf('%d',nx);
   fprintf('%e',err);
   end
end