%% Overborrowing and Systemic Externalities over the Business Cycle
%   Javier Bianchi
%
%   by Damian Romero

% Housekeeping
clear
close all
clc

% Calibration
r       = 4/100;            % Interest rate
sigma   = 2;                % Risk aversion
eta     = (1-0.83)/0.83;    % Elasticity of substitution
kappa   = 0.32;             % Credit coefficient
omega   = 0.31;             % Weight of tradables on CES
betta   = 0.91;             % Discount factor

% Stochastic processes
rho     = [ 0.901 0.495;...
            -0.453 0.225];  % Correlation matrix
Vs      = [ 0.219 0.162;        
            0.162 0.167]./100; % Variance-covariance matrix
NT      = 2;                % Number of grid points for tradables
NN      = 2;                % Number of grid points for nontradables
muT     = 0;                % Mean of tradables
muN     = 0;                % Mean of nontradables
rhoT    = 0.53;             % First-order autocorrelation of tradables
rhoN    = 0.61;             % First-order autocorrelation of nontradables
sigmaT  = 0.058;            % Standard deviation of tradables
sigmaN  = 0.057;            % Standard deviation of nontradables

[yN,PN] = tauchen(NN,muN,rhoN,sigmaN,3);
[yT,PT] = tauchen(NT,muT,rhoT,sigmaT,3);

T       = repmat(PT,NN,NN).*kron(PN,ones(NT,NT));

% % Tradables
% [zT,PT] = tauchenhussey(NT,muT,rhoT,sigmaT,sigmaT);
% 
% % Nontradables
% [zN,PN] = tauchenhussey(NN,muN,rhoN,sigmaN,sigmaN);
% 
% % Joint process (T moves faster--first column--then N)
% [g,T] = var_markov(zT,PT,zN,PN,Vs);

% The VAR(1) has order [yT yN]. Terry's function orders as first row of g
% (not g') as the first variable in the VAR (i.e. yT) and so on. In this
% case, yT is moving slowly related to yN (different way of ordering states
% as usual, where the first column moves faster than the following ones).
% rng('default')
% [T,g] = fn_VAR_to_markov(eye(2),[0;0],rho,Vs,[NT;NN],1000,1);g=g'; % Terry

% Grid for bonds (aggregate)
NB      = 20;               % Number of nodes for aggregate debt
bmin    = -1.11; 
bmax    = -0.31;
B       = linspace(bmin,bmax,NB)';

% Constructing matrices as NShock x Nbonds
b       = repmat(B,1,NT*NN)';
bp      = b;
price   = ones(NT*NN,NB);
c       = ones(NT*NN,NB);
cbind   = c;

% Define fspace and interpolation matrix (interpolation scheme)
fspace  = fundefn('spli',NB,bmin,bmax,1);
Phi     = funbas(fspace);

% Final grid (yT,yN,B)
grid = gridmake(yT,yN,B);%grid    = gridmake(g,B);
yT      = exp(grid(:,1)); yT = reshape(yT,NT*NN,NB);
yN      = exp(grid(:,2)); yN = reshape(yN,NT*NN,NB);

% Technical parameters
tol     = 1e-5;             % Tolerance
uptd    = 0.2;              % Value function updating
err1    = 1;                % Initial error
outfreq = 20;

% Iteration
emu = zeros(NT*NN,NB);
emu2= zeros(NT*NN,NB);
U   = zeros(NT*NN,NB);

bpmax = -kappa*(price.*yN+yT);

options = optimoptions('fsolve','Display','off');
disp('DE Iter      Norm');
iter = 0;
tic
while err1>tol
    
    % Saving old values
    oldp    = price;
    oldc    = c;
    oldbp   = bp;
    
    totalc  = omega*c.^(-eta)+(1-omega)*yN.^(-eta);
    mup     = real(omega*(totalc.^(sigma/eta)).*(totalc.^(-1/eta-1)).*(c.^(-eta-1)));  %mup(y,b)

    % Expected marginal utility in today's grid
    for i=1:NB
        for j=1:NT*NN
            %emu(j,i) = betta*(1+r)*funeval(Phi\mup',fspace,bp(j,i))*T(j,:)';
            emu(j,i) = betta*(1+r)*interp1(B,mup',bp(j,i),'linear','extrap')*T(j,:)';
        end
    end
    
%     for ii=1:NB
%         for jj=1:NT*NN
%             for kk=1:NT*NN
%                 aux(jj,ii,kk)=betta*(1+r)*interp1(B,mup(kk,:)',bp(jj,ii),'linear')*T(jj,kk);
%             end
%         end
%     end
%     emu2=sum(aux,3);
%     asdf=emu2-emu; if max(abs(asdf(:)))>tol;warning('problem!'),end
        
%     for j=1:NT*NN
%         emu(j,:) = (betta*(1+r)*interp1(B,mup',bp(j,:),'linear','extrap')*T(j,:)')';
%     end
    
    % Solve the system of equations
    for i=1:NB
        for j=1:NT*NN
            % Compute U (see appendix of the paper)
            U(j,i) = (omega*(cbind(j,i)^-eta)+(1-omega)*(yN(j,i)^(-eta)))^(sigma/eta-1/eta-1)*omega*(cbind(j,i)^(-eta-1))-emu(j,i);
            
            % Check Lagrange multiplier            
            if U(j,i)>tol  % Credit constraint binds              
                bp(j,i) = bpmax(j,i);
                c(j,i)  = cbind(j,i);
            else % Credit constraint does not bind
                f = @(cc) (omega*(cc^(-eta))+(1-omega)*(yN(j,i)^(-eta)))^(sigma/eta-1/eta-1)*omega*(cc^(-eta-1))-emu(j,i);
                %[c(j,i),U(j,i)] = fsolve(f,0.001,options);
                [c(j,i),U(j,i)] = broyden(f,0.001);
                bp(j,i) = min(max(b(j,i)*(1+r)+yT(j,i)-c(j,i),bmin),bmax);
                fprintf('pp')
            end
        end
    end
    
    % Compute price
    price = real(((1-omega)/omega)*(c./yN).^(1+eta));
    
    % Check collateral constraint
    c = (1+r)*b+yT-max(bp,-kappa*(price.*yN+yT));  %b'>=-kappa*(yt+yn*pn)
    price = real(((1-omega)/omega)*(c./yN).^(1+eta));
    bp = (1+r)*b+yT-c;
    
    % Updating rules
    bp      = uptd*bp+(1-uptd)*oldbp;
    c       = uptd*c+(1-uptd)*oldc;
    price   = uptd*price+(1-uptd)*oldp;
    
    bpmax = -kappa*(price.*yN+yT);
    bpmax(bpmax>bmax) = bmax;
    bpmax(bpmax<bmin) = bmin;
    cbind = (1+r)*b+yT-bpmax;
    
    err1 = max([norm(c-oldc,inf) norm(price-oldp,inf) norm(bp-oldbp,inf)]);
    iter = iter+1;
    
    if mod(iter,outfreq)==0
        fprintf('%d          %1.7f \n',iter,err1);
    end
end
toc