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
V       = [ 0.219 0.162;        
            0.162 0.167]./100; % Variance-covariance matrix
NT      = 3;                % Number of grid points for tradables
NN      = 3;                % Number of grid points for nontradables
muT     = 0;                % Mean of tradables
muN     = 0;                % Mean of nontradables
rhoT    = 0.53;             % First-order autocorrelation of tradables
rhoN    = 0.61;             % First-order autocorrelation of nontradables
sigmaT  = 0.058;            % Standard deviation of tradables
sigmaN  = 0.057;            % Standard deviation of nontradables

% wT      = 0.5+rhoT/4;
% wN      = 0.5+rhoN/4;
% sigmaZT = sigma_eps/sqrt(1-rho^2);

% Tradables
[zT,PT] = tauchenhussey(NT,muT,rhoT,sigmaT,sigmaT);

% Nontradables
[zN,PN] = tauchenhussey(NN,muN,rhoN,sigmaN,sigmaN);

% Joint process (T moves faster--first column--then N)
[g,T] = var_markov(zT,PT,zN,PN);

% Parameters
pars = [r,sigma,eta,kappa,omega,betta];

% Grid for bonds (aggregate)
NB      = 20;               % Number of nodes for aggregate debt
bmin    = -1.11; 
bmax    = -0.31;
B       = linspace(bmin,bmax,NB)';
b       = repmat(B',NN*NT,1); % Bonds-state

% Final grid (yT,yN,B)
grid    = gridmake(g,B);
yT      = exp(grid(:,1)); yT = reshape(yT,NT*NN,NB);
yN      = exp(grid(:,2)); yN = reshape(yN,NT*NN,NB);

% Guesses
bpK     = b;
cTK     = r*b+yT;
cK      = (omega*(cTK.^-eta)+(1-omega)*(yN.^-eta)).^(-1/eta);
lambdaK = omega*(cK.^(1+eta-sigma)).*(cTK.^(-eta-1));
pNK     = real(((1-omega)/omega)*((cTK./yN).^(eta+1)));

% Maximum amount of investment in bonds (ensures that tradable consumption
% is not negative in the worst case scenario) [CHECK IF NEEDED TO ADJUST]
bpK_max = (1+r)*b+yT;
bpK_max = min(bpK_max,bmax);

% Technical parameters
tol     = 1e-5;
err     = 1;
uptd    = .1;
outfreq = 20;
options = optimoptions('fsolve','Display','off');

% Pre-allocation
emu     = zeros(NN*NT,NB);
U       = zeros(NN*NT,NB);

% Iteration
iter = 0;
while err>tol
    % Check/solve Euler equation
    for i=1:NB
        for j=1:NN*NT
            if eulereq(bmin,b(i),B,yT(j),yN(j),lambdaK,pars,T(j,:))>=0
                bpK(j,i) = bmin;
            elseif eulereq(bpK_max(j,i),b(i),B,yT(j),yN(j),lambdaK,pars,T(j,:))<=0
                bpK(j,i) = bpK_max(j,i);
            else
                %bpK(j,i) = fzero('eulereq',[bmin bpK_max(j,i)],[],b(i),B,yT(j),yN(j),lambdaK,pars,T(j,:));
                bpK(j,i) = bisect('eulereq',bmin,bpK_max(j,i),b(i),B,yT(j),yN(j),lambdaK,pars,T(j,:));
            end
        end
    end
    
    % Check collateral constraint
    bpK = max(bpK,-kappa*(yT+pNK.*yN));
    
    % Consumption, marginal utility and prices
    cTK_n       = yT+(1+r)*b-bpK;
    c           = (omega*(cTK_n.^-eta)+(1-omega)*(yN.^-eta)).^(-1/eta);
    lambda_n    = omega*(c.^(1+eta-sigma)).*(cTK_n.^(-eta-1));
    pN_n        = real(((1-omega)/omega)*((cTK_n./yN).^(eta+1)));
    
    % Check convergence
    err = max([norm(pNK-pN_n,inf) norm(bpK-b,inf) norm(cTK_n-cTK,inf)]);
    
    % Update
    cTK = uptd*cTK_n+(1-uptd)*cTK;
    pNK = uptd*pN_n+(1-uptd)*pNK;
    b   = uptd*bpK+(1-uptd)*b;
    
    if mod(iter,outfreq)==0
        fprintf('%d          %1.7f \n',iter,err);
    end
    iter = iter+1;
    
end
% while err>tol
%     
%     % 0) Save old values
%     oldP = pNK;
%     oldC = cTK;
%     oldB = bpK;
%     
%     % 1) Solve for new values [MAYBE OUTSIDE]
%     % 1.1) Bonds and tradable consumption
%     bpKn    = -kappa*(pNK.*yN+yT);
%     cTn     = (1+r)*b+yT-bpKn;
%     
%     % 1.2) Compute Euler equation and check constraint
%     auxC    = omega*cTK.^(-eta)+(1-omega)*yN.^(-eta);
%     mup     = real(omega*(auxC.^(sigma/eta)).*(auxC.^(-1/eta-1)).*(cTK.^(-eta-1)));
%     for i=1:NB
%         for j=1:NN*NT
%             
%             % Compute Euler
%             emu(j,i)    = betta*(1+r)*interp1(B,mup',bpKn(j,i),'linear','extrap')*T(j,:)';
%             U(j,i)      = mup(j,i)-emu(j,i);
%             
%             % Check constraint
%             if U(j,i)>tol
%                 bpKn(j,i)   = bpK_max(j,i);
%             else
%                 f           = @(cc) (omega*(cc^(-eta))+(1-omega)*...
%                     (yT(j,i)))^(sigma/eta-1/eta-1)*omega*(cc^(-eta-1))-emu(j,i);
%                 cTn(j,i)    = fsolve(f,0.001,options);
%                 bpKn(j,i)   = max(min(b(j,i)*(1+r)+yT(j,i)-cTn(j,i),bmax),bmin);
%             end
%             
%         end
%     end
%     
%     % 2) Solve for price
%     pNKn = real(((1-omega)/omega)*((cTK./yN).^(eta+1)));
%     
%     a = 1;
%     
% end