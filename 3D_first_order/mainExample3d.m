% 1st order
clc; clear; close all
tic
format long

%% parameter settings
tol = 0.001;
kmax = 2000;
N = 65;
x = linspace(-1, 1, N);
y = linspace(-1, 1, N);
z = linspace(-1, 1, N);
%%% !!!!!!!!!!!!!!!!!!! Here we use ndgrid instead of meshgrid 
[X, Y, Z] =ndgrid(x,y, z);

%% examples
c_exact=3-0.5*exp(-4*((X).^2+(Y+0.5).^2+(Z).^2))-exp(-4*((X).^2+(Y-0.25).^2+(Z).^2));

%% source point

if 1
snum=18;
sourceset=cell(18,1);
for i=1:9
    [ii,jj]=ind2sub([3,3],i);
    sourceset{i}=[-1+ii*32/(N-1),-1+jj*32/(N-1),-0.75];

end
for i=1:9
    [ii,jj]=ind2sub([3,3],i);
    sourceset{9+i}=[-1+ii*32/(N-1),-1+jj*32/(N-1),0.75];
end
end
if 0

snum=98;
sourceset=cell(snum,1);
for i=1:49
    [ii,jj]=ind2sub([7,7],i);
    sourceset{i}=[-1+ii*8/(N-1),-1+jj*8/(N-1),-0.75];

end
for i=1:49
    [ii,jj]=ind2sub([7,7],i);
    sourceset{49+i}=[-1+ii*8/(N-1),-1+jj*8/(N-1),0.75];
end
end


%% iteration
I = N;
J = N;
K = N;
dx = (x(end)-x(1)) / (I-1); dy = (y(end) - y(1)) / (J-1); dz = (z(end) - z(1)) / (K-1);
h=dx;
niu = 1;
c0 = poisson3d_fd_fullbc(zeros(I-2,J-2,K-2), c_exact, h, niu);

c = c0;

energy = 1e9;
 alpha_f = 1e-4; alpha_0 = 0.1;


T_star=cell(length(sourceset));

parfor kk = 1:length(sourceset)
T_star{kk}=Eikonal_3d_1st([-1,1],[-1,1],[-1,1],sourceset{kk},h,1./c_exact.^2);
end

for k = 1: kmax
    
    energy_p = 0;
    cstar = 0;
    parfor p_num = 1:length(sourceset)
        T =  Eikonal_3d_1st([-1,1],[-1,1],[-1,1],sourceset{p_num},h,1./c.^2);
       
        energy_p = energy_p + EnergyFun3D(T, T_star{p_num}, dx, dy, dz);
        
        %beta = lambda_3d(T, T_star{p_num}, dx, dy,dz);
        
        % use T^2 or not 
        squareoption=1;
        
        beta = beta_solver_3d_ndgrid(T, T_star{p_num}, dx, dy, dz,squareoption)
        
         ctemp = -beta ./ c.^3;
         ctemp2=0*ctemp;
       ctemp2(2:end-1,2:end-1,2:end-1)= poisson_fft3(ctemp(2:end-1,2:end-1,2:end-1), dx, dy, dz, niu);
        cstar = cstar + ctemp2;
    end
    
    if energy_p < tol
        break
    end

    energy = [energy, energy_p];
    if mod(k, 10) == 0
       disp(k)
       disp(energy(k+1))
    end
     cerror(k)=sum(abs((c(:)-c_exact(:))))*dx*dy*dz;
     if 1
     energy_p
     sum(abs((c(:)-c_exact(:))))*dx*dy*dz
     end

     %         alpha = 0.01;
    % alpha = alpha_f + 0.5*(alpha_0 - alpha_f) * (1 + cos(pi*k / kmax));
     alpha=1;
    c = c + alpha * cstar;
   
end

