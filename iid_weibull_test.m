clear all

num = [0.6,0.4,0.3, 0.2];

for jj = 1:length(num)
    set(gca,'fontsize',18)
    subplot(2,2,jj)

R = 10;
n=1e4; 
p_prop = 0.05; % proportion to use to estimate p_star
n_pstar = ceil(p_prop * n); % number of samples to use to estimate p_star

AS = zeros(R,1);
Prop = zeros(R,1); % proposed method (Push-Out)
AK = zeros(R,1);

num(jj)
for rr = 1:R

%Weibull Example
%note: b=1 corresponds to sum of Exp(lambda) RVS 
%b controls the tail heaviness (smaller --> heavier)

d=30;
lam=1; %scale --- parametrization in the Handbook of Monte Carlo Methods (Kroese et. al.)
b=num(jj); %shape --- alpha in the HMCM 

a=1/lam; %scale for MATLAB's built in functions 

F_inv= @(x) wblinv(x,a,b);
F = @(x) wblcdf(x,a,b);
f = @(x) wblpdf(x,a,b);
tic

mn=a*gamma(1+(1/b))*d;
m=1;
gam=mn*m;



%CMC
X=wblrnd(a,b, n,d); 
S=sum(X,2);
ind=(S<gam);
CDF=mean(ind);

tic 
%Asmussen pdf estimator 
X=wblrnd(a,b,n,d);
S=sum(X,2);

pdf=zeros(n,d);

for i=1:d  
    v=gam-S+X(:,i); 
    pdf(:,i)=f(v);
end    

    est=sum(pdf,2)/d; %single estimate
    ell=mean(est);
    AS(rr) = ell;

    
 %ASMUSSEN KROESE       
X=wblrnd(a,b,n,d);

pdf=zeros(n,d);

for i=1:d  
    tmp_X = X;
    tmp_X(:,i) = [];
    
    M = max(tmp_X,[],2);
    S = sum(tmp_X,2);
    
    ind = (S + M) < gam;
    pdf(:,i)=f(gam - S) .* ind ;
end    

est=sum(pdf,2); %single estimate

ell=mean(est);

RE_a=std(est)/mean(est)/sqrt(n);
    

AK(rr) = ell ;    
 

tic
X=wblrnd(a,b,n,d);
S=sum(X,2); 

c1=b-1;
c2=b*(lam.^b);

Xp=X.^(c1+1);
S_Xp=-c2*sum(Xp,2)+d*(c1+1);

ind_left = (S<gam);
ind_right = (S>=gam);

pdf=(1/gam)*S_Xp.* ind_left;
pdf_RT=-(1/gam)*S_Xp.* ind_right;

est = mean(pdf);
est_RT = mean(pdf_RT);

RE=std(pdf)/mean(pdf)/sqrt(n);
RE_RT=std(pdf_RT)/mean(pdf_RT)/sqrt(n); 

% uses first 5000 for estimating p_star



C = cov(pdf(1:n_pstar), pdf_RT(1:n_pstar));
p_star = (C(2,2)-C(1,2)) / (C(1,1) + C(2,2) - 2 * C(1,2));

pdf = pdf(n_pstar:end);
pdf_RT = pdf_RT(n_pstar:end);

pdf_PushOut=p_star*pdf+ (1-p_star)* pdf_RT;
Prop(rr) = mean(pdf_PushOut);


end
boxplot([AS ,AK,  Prop], 'Labels', {'Conditional', 'AK','Push-Out'})
tt = sprintf('%.1f', num(jj));

title(tt,  'interpreter', 'latex', 'FontSize', 20) 
figure(1)
    set(gca,'fontsize',18)
    set(findobj(gca,'type','line'),'linew',2)
end