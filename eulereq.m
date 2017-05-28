function EE = eulereq(bp,binit,b,yT,yN,lambdap,pars,T)
% Compute the Euler equation of the model

r       = pars(1);
sigma   = pars(2);
eta     = pars(3);
%kappa   = pars(4);
omega   = pars(5);
betta   = pars(6);

cT      = yT+binit*(1+r)-bp;
c       = (omega*(cT.^-eta)+(1-omega)*(yN.^-eta)).^(-1/eta);
lambda  = omega*(c.^(1+eta-sigma)).*(cT.^(-eta-1));
Elambda = interp1(b,lambdap',bp,'linear','extrap')*T';

EE      = lambda-(1+r)*betta*Elambda;