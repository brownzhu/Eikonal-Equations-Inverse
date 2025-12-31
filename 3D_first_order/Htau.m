function ham=Htau(i,j,k,u,p,q,l,sourcex,sourcey,sourcez,s0,h)
r=sqrt((i-sourcex)^2+(j-sourcey)^2+(k-sourcez)^2)*h;
tau0=s0*r;
ham=sqrt(tau0^2*(p^2+q^2+l^2)+2*tau0*u*(s0*(i-sourcex)*h/r*p+s0*(j-sourcey)*h/r*q+s0*(k-sourcez)*h/r*l)+u^2*s0^2);