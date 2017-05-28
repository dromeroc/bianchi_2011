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
NT      = 4;                % Number of grid points for tradables
NN      = 4;                % Number of grid points for nontradables
muT     = 0;                % Mean of tradables
muN     = 0;                % Mean of nontradables
rhoT    = 0.53;             % First-order autocorrelation of tradables
rhoN    = 0.61;             % First-order autocorrelation of nontradables
sigmaT  = 0.058;            % Standard deviation of tradables
sigmaN  = 0.057;            % Standard deviation of nontradables

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
rng('default')
[T,g] = fn_VAR_to_markov(eye(2),[0;0],rho,Vs,[NT;NN],1000,1);g=g'; % Terry

% Grid for bonds (aggregate)
NB      = 80;               % Number of nodes for aggregate debt
bmin    = -1.11; 
bmax    = -0.31;
B       = linspace(bmin,bmax,NB)';

% Constructing matrices as NShock x Nbonds
b       = repmat(B,1,NT*NN)';
bp      = b;
price   = ones(NT*NN,NB);
c       = ones(NT*NN,NB);
cbind   = c;

% Final grid (yT,yN,B)
grid    = gridmake(g,B);
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
while err1>tol || iter<1
    
    % Saving old values
    oldp    = price;
    oldc    = c;
    oldbp   = bp;
    
    totalc  = omega*c.^(-eta)+(1-omega)*yN.^(-eta);
    mup     = real(omega*(totalc.^(sigma/eta)).*(totalc.^(-1/eta-1)).*(c.^(-eta-1)));  %mup(y,b)

    % Expected marginal utility in today's grid
    for i=1:NB
        for j=1:NT*NN
            emu(j,i) = betta*(1+r)*interp1(B,mup',bp(j,i),'linear','extrap')*T(j,:)';
        end
    end
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
%% Social planner

uptd    = uptd/5;
cSP     = c;
priceSP = price;
bpSP    = bp;
cbindSP = c;
bpmaxSP = bpSP;
EESP    = U;

emuSP   = zeros(NT*NN,NB);

iter    = 0;
err2    = 1;
disp('SPP Iter     Norm');
tic
while err2>tol
    
    % Saving old values
    oldpSP    = priceSP;
    oldcSP    = cSP;
    oldbpSP   = bpSP;
    
    totalcSP = omega*cSP.^(-eta)+(1-omega)*yN.^(-eta);
    mupSP = real(omega*(totalcSP.^(sigma/eta)).*(totalcSP.^(-1/eta-1)).*(cSP.^(-eta-1)));  %mup(y,b)
    psi = kappa*(1+eta)*priceSP.*yN./cSP;
    mupSP = mupSP+EESP.*psi;

   % Expected marginal utility in today's grid
    for i=1:NB
        for j=1:NT*NN
            emuSP(j,i) = betta*(1+r)*interp1(B,mupSP',bpSP(j,i),'linear','extrap')*T(j,:)';
        end
    end
    
    % Solve the system of equations
    for i=1:NB
        for j=1:NT*NN
            % Compute U (see appendix of the paper)
            EESP(j,i) = (omega*(cbindSP(j,i)^-eta)+(1-omega)*(yN(j,i)^(-eta)))^(sigma/eta-1/eta-1)*omega*(cbindSP(j,i)^(-eta-1))-emuSP(j,i);
            
            % Check Lagrange multiplier            
            if EESP(j,i)>tol  % Credit constraint binds              
                bpSP(j,i) = bpmaxSP(j,i);
                cSP(j,i)  = cbindSP(j,i);
            else % Credit constraint does not bind
                f = @(cc) (omega*(cc^(-eta))+(1-omega)*(yN(j,i)^(-eta)))^(sigma/eta-1/eta-1)*omega*(cc^(-eta-1))-emuSP(j,i);
                %[cSP(j,i),EESP(j,i)] = fsolve(f,0.001,options);
                [cSP(j,i),EESP(j,i)] = broyden(f,0.001);
                bpSP(j,i) = min(max(b(j,i)*(1+r)+yT(j,i)-cSP(j,i),bmin),bmax);
            end
        end
    end
    
    % Compute price
    priceSP = real(((1-omega)/omega)*(cSP./yN).^(1+eta));
    
    % Check collateral constraint
    cSP = (1+r)*b+yT-max(bpSP,-kappa*(priceSP.*yN+yT));  %b'>=-kappa*(yt+yn*pn)
    priceSP = real(((1-omega)/omega)*(cSP./yN).^(1+eta));
    bpSP = (1+r)*b+yT-cSP;
    
    % Updating rules
    bpSP      = uptd*bpSP+(1-uptd)*oldbpSP;
    cSP       = uptd*cSP+(1-uptd)*oldcSP;
    priceSP   = uptd*priceSP+(1-uptd)*oldpSP;
    
    bpmaxSP = -kappa*(priceSP.*yN+yT);
    bpmaxSP(bpmaxSP>bmax) = bmax;
    bpmaxSP(bpmaxSP<bmin) = bmin;
    cbindSP = (1+r)*b+yT-bpmaxSP;
    
    err2= max([norm(cSP-oldcSP,inf) norm(priceSP-oldpSP,inf) norm(bpSP-oldbpSP,inf)]);
    iter = iter+1;
    
    if mod(iter,outfreq)==0
        fprintf('%d          %1.7f \n',iter,err2);
    end
    
end
toc

% 

%% Welfare

V       = zeros(NN*NT,NB);
VSP     = zeros(NN*NT,NB);
Ut      = ((totalc.^(-1/eta)).^(1-sigma))./(1-sigma);
UtSP    = ((totalcSP.^(-1/eta)).^(1-sigma))./(1-sigma);
EV      = zeros(NN*NT,NB);
EVSP    = zeros(NN*NT,NB);

iter    = 0;
err3    = 1;
disp('Value Iter     Norm');
while err3>tol
    % Computing expected utility
    for i=1:NB
        for j=1:NT*NN
            EV(j,i)     = betta*interp1(B,V',bp(j,i),'linear','extrap')*T(j,:)';
            EVSP(j,i)   = betta*interp1(B,VSP',bpSP(j,i),'linear','extrap')*T(j,:)';
        end
    end
    
    % Computing value function
    V_new   = Ut+EV;
    VSP_new = UtSP+EVSP;
    
    % Checking convergence
    err3    = max([norm((V_new-V)./V,inf) norm((VSP_new-VSP)./VSP,inf)]);
    V       = V_new;
    VSP     = VSP_new;
    iter = iter+1;
    
    if mod(iter, outfreq) == 0
        fprintf('%d          %1.7f \n',iter,err3);
    end
end
Wf = (VSP./V).^(1/(1-sigma))-1;

% plot(B,Wf(1,:)*100)

%% Optimal tax

mupSP_tax = mupSP-EESP.*psi;
for i=1:NB
    for j=1:NN*NT
        emuSP(j,i) = betta*(1+r)*interp1(B,mupSP_tax',bpSP(j,i),'linear','extrap')*T(j,:)';
    end
end
Tao = mupSP_tax./emuSP;

% Tax is set to one when constraint binds
AN = bpSP(:,2:end)-bpSP(:,1:end-1);

for i=1:NN*NT
    IN(i) = find(AN(i,:)>0,1,'first');
end

for j=1:NN*NT
    if IN(j)>1
        Tao(j,1:IN(j)-1) = 1;
    end
end

Tao(Tao<1) = 1;

% plot(B,(Tao(end,:)-1)*100)

%% Simulations

Time    = 200000;
cut     = 1000;  % Initial condition dependence
[S_index,] = markov(T,Time+cut+1,1,1:length(T));

bpSIM=zeros(Time+cut,1);
WfSIM=zeros(Time+cut,1);
bindSIM=zeros(Time+cut,1);
cTSIM=zeros(Time+cut,1);
pSIM=zeros(Time+cut,1);
 
bpSP_SIM=zeros(Time+cut,1);
bindSP_SIM=zeros(Time+cut,1);
cTSP_SIM=zeros(Time+cut,1);
pSP_SIM=zeros(Time+cut,1);
 
Tao_SIM=zeros(Time+cut,1);
Tregion_SIM=zeros(Time+cut,1);
yTSIM=zeros(Time+cut,1);
yNSIM=zeros(Time+cut,1);
RSIM=zeros(Time+cut,1);
 
bpSIM(1)=bmax+(bmin-bmax)/2;
bpSP_SIM(1)=bmax+(bmin-bmax)/2;

for i=2:Time+cut
    bpSIM(i)=interp1(B,bp(S_index(i),:),bpSIM(i-1),'linear');
    WfSIM(i)=interp1(B,Wf(S_index(i),:),bpSIM(i-1),'linear');
    yTSIM(i)=yT(S_index(i)); 
    yNSIM(i)=yN(S_index(i)); 
%     RSIM(i)=R(1,S_index(i));  
    cTSIM(i)=(1+r)*bpSIM(i-1)+yTSIM(i)-bpSIM(i); 
    bpSP_SIM(i)=interp1(B,bpSP(S_index(i),:),bpSP_SIM(i-1),'linear');
    cTSP_SIM(i)=(1+r)*bpSP_SIM(i-1)+yTSIM(i)-bpSP_SIM(i); 
    Tao_SIM(i)=interp1(B,Tao(S_index(i),:),bpSP_SIM(i-1),'linear');
end

% Eliminate Initial Condition Dependency
S_index=S_index(cut+1:Time+cut)';

bpSIM=bpSIM(cut+1:Time+cut);
bindSIM=bindSIM(cut+1:Time+cut);
%bplim_SIM=bplim_SIM(cut+1:Time+cut);
cTSIM=cTSIM(cut+1:Time+cut);
pSIM=pSIM(cut+1:Time+cut);
%CA_SIM=CA_SIM(cut:Time+cut-1);

bpSP_SIM=bpSP_SIM(cut+1:Time+cut);
bindSP_SIM=bindSP_SIM(cut+1:Time+cut);
%bplimSP_SIM=bplimSP_SIM(cut+1:Time+cut);
cTSP_SIM=cTSP_SIM(cut+1:Time+cut);
pSP_SIM=pSP_SIM(cut+1:Time+cut);
%CASP_SIM=CASP_SIM(cut:Time+cut-1);

WfSIM=WfSIM(cut+1:Time+cut);
Tao_SIM=Tao_SIM(cut+1:Time+cut);
Tregion_SIM=Tregion_SIM(cut+1:Time+cut);
yTSIM=yTSIM(cut+1:Time+cut);
yNSIM=yNSIM(cut+1:Time+cut);
RSIM=RSIM(cut+1:Time+cut);

cSIM    = (omega*(cTSIM.^-eta)+(1-omega)*(yNSIM.^-eta)).^(-1/eta);
cSIMd   = 100*(cSIM./mean(cSIM)-1);
cSP_SIM = (omega*(cTSP_SIM.^-eta)+(1-omega)*(yNSIM.^-eta)).^(-1/eta);
cSP_SIMd= 100*(cSP_SIM./mean(cSP_SIM)-1);

% Plots
subplot(1,2,1)
h1 = hist(bpSP_SIM,bmin:0.001:bmax);
hist(bpSIM,bmin:0.001:bmax),hold on,plot(bmin:0.001:bmax,h1,'r','LineWidth',1.5)
legend('Decentralized equilibrium','Social planner')

subplot(1,2,2)
h2 = hist(cSP_SIM,0:0.001:1.1876);
hist(cSIM,0:0.001:1.1876),hold on,plot(0:0.001:1.1876,h2,'r','LineWidth',1.5)
legend('Decentralized equilibrium','Social planner')