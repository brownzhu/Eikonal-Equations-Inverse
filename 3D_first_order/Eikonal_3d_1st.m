function [tau]=Eikonal_3d_1st(computex,computey,computez,source,h,n2)
tic
epsilon=1e-6;
example=1;% nu=(3-1.75*exp(-((x-1)^2+(y-1)^2+(z-1)^2)/0.64))^2; n2=4;
nx=round((computex(2)-computex(1))/h)+1;
ny=round((computey(2)-computey(1))/h)+1;
nz=round((computez(2)-computez(1))/h)+1;
sourcex=round((source(1)-computex(1))/h)+1;
sourcey=round((source(2)-computey(1))/h)+1;
sourcez=round((source(3)-computez(1))/h)+1;
num=1.2;

%% first order initialization for Tau
S0=n2(sourcex,sourcey,sourcez);
M = 20;
Tau=zeros(nx,ny,nz)+M;

Tau(sourcex,sourcey,sourcez)=0;
flag=zeros(nx,ny,nz)+1;
flag(sourcex,sourcey,sourcez)=0;

[YY,XX,ZZ]=meshgrid(computey(1):h:computey(2),computex(1):h:computex(2),computez(1):h:computez(2));
p0=sqrt(S0)*(XX-source(1))./sqrt((XX-source(1)).^2+(YY-source(2)).^2+(ZZ-source(3)).^2);
q0=sqrt(S0)*(YY-source(2))./sqrt((XX-source(1)).^2+(YY-source(2)).^2+(ZZ-source(3)).^2);
r0=sqrt(S0)*(ZZ-source(3))./sqrt((XX-source(1)).^2+(YY-source(2)).^2+(ZZ-source(3)).^2);
Tau0=sqrt(S0)*sqrt((XX-source(1)).^2+(YY-source(2)).^2+(ZZ-source(3)).^2);
U=Tau./Tau0;
U(sourcex,sourcey,sourcez)=1;
F=sqrt(n2);
% ax=2*(areax(2)-areax(1));
% ay=2*(areay(2)-areay(1));
% az=2*(areaz(2)-areaz(1));

%%

iter=0;
while iter <3000
    Usave=U;
    iter;
    for i =2:nx-1
        for j = 2 : ny-1
            for k = 2: nz -1
                uxp=(U(i+1,j,k)-U(i,j,k))/h;
                uxm=(U(i,j,k)-U(i-1,j,k))/h;
                uyp=(U(i,j+1,k)-U(i,j,k))/h;
                uym=(U(i,j,k)-U(i,j-1,k))/h;
                uzp=(U(i,j,k+1)-U(i,j,k))/h;
                uzm=(U(i,j,k)-U(i,j,k-1))/h;
                ax=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                ay=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                az=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));                
                unew=1/(ax/h+ay/h+az/h)*(F(i,j,k)-Htau(i,j,k,U(i,j,k),...
                    (uxp+uxm)/2,(uyp+uym)/2,(uzp+uzm)/2,sourcex,sourcey,...
                     sourcez,sqrt(S0),h)+ax/2/h*(2*U(i,j,k)+(uxp-uxm)*h)+ay/2/h*(2*U(i,j,k)...
                     +(uyp-uym)*h)+az/2/h*(2*U(i,j,k)+(uzp-uzm)*h));
                 if flag(i,j,k)==1
                     U(i,j,k)=min(U(i,j,k),unew);
                 end
            end
        end
    end
    U(1,:,:)=min(U(1,:,:),2*U(2,:,:)-U(3,:,:));
    U(:,1,:)=min(U(:,1,:),2*U(:,2,:)-U(:,3,:));
    U(:,:,1)=min(U(:,:,1),2*U(:,:,2)-U(:,:,3));
    U(end,:,:)=min(U(end,:,:),2*U(end-1,:,:)-U(end-2,:,:));
    U(:,end,:)=min(U(:,end,:),2*U(:,end-1,:)-U(:,end-2,:));
    U(:,:,end)=min(U(:,:,end),2*U(:,:,end-1)-U(:,:,end-2));
%  
    for i =nx-1:-1:2
        for j =  ny-1:-1:2
            for k = nz -1:-1:2
                uxp=(U(i+1,j,k)-U(i,j,k))/h;
                uxm=(U(i,j,k)-U(i-1,j,k))/h;
                uyp=(U(i,j+1,k)-U(i,j,k))/h;
                uym=(U(i,j,k)-U(i,j-1,k))/h;
                uzp=(U(i,j,k+1)-U(i,j,k))/h;
                uzm=(U(i,j,k)-U(i,j,k-1))/h;
                                ax=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                ay=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                az=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                unew=1/(ax/h+ay/h+az/h)*(F(i,j,k)-Htau(i,j,k,U(i,j,k),...
                    (uxp+uxm)/2,(uyp+uym)/2,(uzp+uzm)/2,sourcex,sourcey,...
                     sourcez,sqrt(S0),h)+ax/2/h*(2*U(i,j,k)+(uxp-uxm)*h)+ay/2/h*(2*U(i,j,k)...
                     +(uyp-uym)*h)+az/2/h*(2*U(i,j,k)+(uzp-uzm)*h));
                 if flag(i,j,k)==1
                     U(i,j,k)=min(U(i,j,k),unew);
                 end
            end
        end
    end
    U(1,:,:)=min(U(1,:,:),2*U(2,:,:)-U(3,:,:));
    U(:,1,:)=min(U(:,1,:),2*U(:,2,:)-U(:,3,:));
    U(:,:,1)=min(U(:,:,1),2*U(:,:,2)-U(:,:,3));
    U(end,:,:)=min(U(end,:,:),2*U(end-1,:,:)-U(end-2,:,:));
    U(:,end,:)=min(U(:,end,:),2*U(:,end-1,:)-U(:,end-2,:));
    U(:,:,end)=min(U(:,:,end),2*U(:,:,end-1)-U(:,:,end-2));
    
    
    for i =2:nx-1
        for j =   ny-1:-1:2
            for k = 2: nz -1
                uxp=(U(i+1,j,k)-U(i,j,k))/h;
                uxm=(U(i,j,k)-U(i-1,j,k))/h;
                uyp=(U(i,j+1,k)-U(i,j,k))/h;
                uym=(U(i,j,k)-U(i,j-1,k))/h;
                uzp=(U(i,j,k+1)-U(i,j,k))/h;
                uzm=(U(i,j,k)-U(i,j,k-1))/h;
                                ax=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                ay=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                az=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                unew=1/(ax/h+ay/h+az/h)*(F(i,j,k)-Htau(i,j,k,U(i,j,k),...
                    (uxp+uxm)/2,(uyp+uym)/2,(uzp+uzm)/2,sourcex,sourcey,...
                     sourcez,sqrt(S0),h)+ax/2/h*(2*U(i,j,k)+(uxp-uxm)*h)+ay/2/h*(2*U(i,j,k)...
                     +(uyp-uym)*h)+az/2/h*(2*U(i,j,k)+(uzp-uzm)*h));
                 if flag(i,j,k)==1
                     U(i,j,k)=min(U(i,j,k),unew);
                 end
            end
        end
    end
    U(1,:,:)=min(U(1,:,:),2*U(2,:,:)-U(3,:,:));
    U(:,1,:)=min(U(:,1,:),2*U(:,2,:)-U(:,3,:));
    U(:,:,1)=min(U(:,:,1),2*U(:,:,2)-U(:,:,3));
    U(end,:,:)=min(U(end,:,:),2*U(end-1,:,:)-U(end-2,:,:));
    U(:,end,:)=min(U(:,end,:),2*U(:,end-1,:)-U(:,end-2,:));
    U(:,:,end)=min(U(:,:,end),2*U(:,:,end-1)-U(:,:,end-2));

    for i =nx-1:-1:2
        for j = 2: ny-1
            for k =  nz -1:-1:2
                uxp=(U(i+1,j,k)-U(i,j,k))/h;
                uxm=(U(i,j,k)-U(i-1,j,k))/h;
                uyp=(U(i,j+1,k)-U(i,j,k))/h;
                uym=(U(i,j,k)-U(i,j-1,k))/h;
                uzp=(U(i,j,k+1)-U(i,j,k))/h;
                uzm=(U(i,j,k)-U(i,j,k-1))/h;
                                ax=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                ay=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                az=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                unew=1/(ax/h+ay/h+az/h)*(F(i,j,k)-Htau(i,j,k,U(i,j,k),...
                    (uxp+uxm)/2,(uyp+uym)/2,(uzp+uzm)/2,sourcex,sourcey,...
                     sourcez,sqrt(S0),h)+ax/2/h*(2*U(i,j,k)+(uxp-uxm)*h)+ay/2/h*(2*U(i,j,k)...
                     +(uyp-uym)*h)+az/2/h*(2*U(i,j,k)+(uzp-uzm)*h));
                 if flag(i,j,k)==1
                     U(i,j,k)=min(U(i,j,k),unew);
                 end
            end
        end
    end
    U(1,:,:)=min(U(1,:,:),2*U(2,:,:)-U(3,:,:));
    U(:,1,:)=min(U(:,1,:),2*U(:,2,:)-U(:,3,:));
    U(:,:,1)=min(U(:,:,1),2*U(:,:,2)-U(:,:,3));
    U(end,:,:)=min(U(end,:,:),2*U(end-1,:,:)-U(end-2,:,:));
    U(:,end,:)=min(U(:,end,:),2*U(:,end-1,:)-U(:,end-2,:));
    U(:,:,end)=min(U(:,:,end),2*U(:,:,end-1)-U(:,:,end-2));
    
    for i =2:nx-1
        for j =  ny-1:-1:2
            for k = nz -1:-1:2
                uxp=(U(i+1,j,k)-U(i,j,k))/h;
                uxm=(U(i,j,k)-U(i-1,j,k))/h;
                uyp=(U(i,j+1,k)-U(i,j,k))/h;
                uym=(U(i,j,k)-U(i,j-1,k))/h;
                uzp=(U(i,j,k+1)-U(i,j,k))/h;
                uzm=(U(i,j,k)-U(i,j,k-1))/h;
                                ax=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                ay=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                az=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                unew=1/(ax/h+ay/h+az/h)*(F(i,j,k)-Htau(i,j,k,U(i,j,k),...
                    (uxp+uxm)/2,(uyp+uym)/2,(uzp+uzm)/2,sourcex,sourcey,...
                     sourcez,sqrt(S0),h)+ax/2/h*(2*U(i,j,k)+(uxp-uxm)*h)+ay/2/h*(2*U(i,j,k)...
                     +(uyp-uym)*h)+az/2/h*(2*U(i,j,k)+(uzp-uzm)*h));
                 if flag(i,j,k)==1
                     U(i,j,k)=min(U(i,j,k),unew);
                 end
            end
        end
    end
    U(1,:,:)=min(U(1,:,:),2*U(2,:,:)-U(3,:,:));
    U(:,1,:)=min(U(:,1,:),2*U(:,2,:)-U(:,3,:));
    U(:,:,1)=min(U(:,:,1),2*U(:,:,2)-U(:,:,3));
    U(end,:,:)=min(U(end,:,:),2*U(end-1,:,:)-U(end-2,:,:));
    U(:,end,:)=min(U(:,end,:),2*U(:,end-1,:)-U(:,end-2,:));
    U(:,:,end)=min(U(:,:,end),2*U(:,:,end-1)-U(:,:,end-2));

    for i =nx-1:-1:2
        for j = 2 : ny-1
            for k = 2: nz -1
                uxp=(U(i+1,j,k)-U(i,j,k))/h;
                uxm=(U(i,j,k)-U(i-1,j,k))/h;
                uyp=(U(i,j+1,k)-U(i,j,k))/h;
                uym=(U(i,j,k)-U(i,j-1,k))/h;
                uzp=(U(i,j,k+1)-U(i,j,k))/h;
                uzm=(U(i,j,k)-U(i,j,k-1))/h;
                                ax=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                ay=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                az=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                unew=1/(ax/h+ay/h+az/h)*(F(i,j,k)-Htau(i,j,k,U(i,j,k),...
                    (uxp+uxm)/2,(uyp+uym)/2,(uzp+uzm)/2,sourcex,sourcey,...
                     sourcez,sqrt(S0),h)+ax/2/h*(2*U(i,j,k)+(uxp-uxm)*h)+ay/2/h*(2*U(i,j,k)...
                     +(uyp-uym)*h)+az/2/h*(2*U(i,j,k)+(uzp-uzm)*h));
                 if flag(i,j,k)==1
                     U(i,j,k)=min(U(i,j,k),unew);
                 end
            end
        end
    end
    U(1,:,:)=min(U(1,:,:),2*U(2,:,:)-U(3,:,:));
    U(:,1,:)=min(U(:,1,:),2*U(:,2,:)-U(:,3,:));
    U(:,:,1)=min(U(:,:,1),2*U(:,:,2)-U(:,:,3));
    U(end,:,:)=min(U(end,:,:),2*U(end-1,:,:)-U(end-2,:,:));
    U(:,end,:)=min(U(:,end,:),2*U(:,end-1,:)-U(:,end-2,:));
    U(:,:,end)=min(U(:,:,end),2*U(:,:,end-1)-U(:,:,end-2));
  
    
    for i =2:nx-1
        for j = 2 : ny-1
            for k =  nz -1:-1:2
                uxp=(U(i+1,j,k)-U(i,j,k))/h;
                uxm=(U(i,j,k)-U(i-1,j,k))/h;
                uyp=(U(i,j+1,k)-U(i,j,k))/h;
                uym=(U(i,j,k)-U(i,j-1,k))/h;
                uzp=(U(i,j,k+1)-U(i,j,k))/h;
                uzm=(U(i,j,k)-U(i,j,k-1))/h;
                                ax=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                ay=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                az=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                unew=1/(ax/h+ay/h+az/h)*(F(i,j,k)-Htau(i,j,k,U(i,j,k),...
                    (uxp+uxm)/2,(uyp+uym)/2,(uzp+uzm)/2,sourcex,sourcey,...
                     sourcez,sqrt(S0),h)+ax/2/h*(2*U(i,j,k)+(uxp-uxm)*h)+ay/2/h*(2*U(i,j,k)...
                     +(uyp-uym)*h)+az/2/h*(2*U(i,j,k)+(uzp-uzm)*h));
                 if flag(i,j,k)==1
                     U(i,j,k)=min(U(i,j,k),unew);
                 end
            end
        end
    end
    U(1,:,:)=min(U(1,:,:),2*U(2,:,:)-U(3,:,:));
    U(:,1,:)=min(U(:,1,:),2*U(:,2,:)-U(:,3,:));
    U(:,:,1)=min(U(:,:,1),2*U(:,:,2)-U(:,:,3));
    U(end,:,:)=min(U(end,:,:),2*U(end-1,:,:)-U(end-2,:,:));
    U(:,end,:)=min(U(:,end,:),2*U(:,end-1,:)-U(:,end-2,:));
    U(:,:,end)=min(U(:,:,end),2*U(:,:,end-1)-U(:,:,end-2));
    
  
    for i =nx-1:-1:2
        for j = ny-1:-1:2
            for k = 2: nz -1
                uxp=(U(i+1,j,k)-U(i,j,k))/h;
                uxm=(U(i,j,k)-U(i-1,j,k))/h;
                uyp=(U(i,j+1,k)-U(i,j,k))/h;
                uym=(U(i,j,k)-U(i,j-1,k))/h;
                uzp=(U(i,j,k+1)-U(i,j,k))/h;
                uzm=(U(i,j,k)-U(i,j,k-1))/h;
                ax=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                ay=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                az=1e-6+num*(abs(Tau0(i,j,k))+h*(abs(p0(i,j,k)+abs(q0(i,j,k))+abs(r0(i,j,k)))));
                unew=1/(ax/h+ay/h+az/h)*(F(i,j,k)-Htau(i,j,k,U(i,j,k),...
                    (uxp+uxm)/2,(uyp+uym)/2,(uzp+uzm)/2,sourcex,sourcey,...
                     sourcez,sqrt(S0),h)+ax/2/h*(2*U(i,j,k)+(uxp-uxm)*h)+ay/2/h*(2*U(i,j,k)...
                     +(uyp-uym)*h)+az/2/h*(2*U(i,j,k)+(uzp-uzm)*h));
                 if flag(i,j,k)==1
                     U(i,j,k)=min(U(i,j,k),unew);
                 end
            end
        end
    end
    U(1,:,:)=min(U(1,:,:),2*U(2,:,:)-U(3,:,:));
    U(:,1,:)=min(U(:,1,:),2*U(:,2,:)-U(:,3,:));
    U(:,:,1)=min(U(:,:,1),2*U(:,:,2)-U(:,:,3));
    U(end,:,:)=min(U(end,:,:),2*U(end-1,:,:)-U(end-2,:,:));
    U(:,end,:)=min(U(:,end,:),2*U(:,end-1,:)-U(:,end-2,:));
    U(:,:,end)=min(U(:,:,end),2*U(:,:,end-1)-U(:,:,end-2));
    

    
max(max(max(abs(U-Usave))));
    if max(max(max(abs(U-Usave))))<epsilon
        break
    else
        Usave=U;
    end
    iter=iter+1;
end

tau=U.*Tau0;

