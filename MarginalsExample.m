function PushOutLR
% -------------------------------------------------------
Z = csvread('pima.csv',1,1);
Y = Z(:, end);
X = Z(:,1:(end-1));

%X = X(:,[1,2]); 
%X = X(:,[1,2,5,6]); % choose covariates of interest
X = X(:,[1,2,5,6,7]);


d = size(X,2) + 1;
X = [ones(size(X,1),1), X]; %create design matrix (add ones)

% standardization
for i = 2:size(X,2)
    X(:,i) = X(:,i) - mean(X(:,i));
    X(:,i) = X(:,i) ./ std(X(:,i));
end
tic

tau = 1; 
%-------------------------------------------------------

rng(12345) % note similar results for rng(123) and rng(1234)

% main run 
burnin = 1e3;
N = 5e4 + burnin; 
thin = 1;

samples = Metropolis(X,Y,tau,d, burnin,N,thin);
    
    gr = zeros(d, size(samples,1));

    for i = 1:size(samples,1)
        gr(:,i) = gradLogPosterior(samples(i,:),X,Y,tau);
    end
 
% longer run 
burnin2 = 1e4;
N2 = 5e6 + burnin2; 
%thin2 = 50;

%long_run = Metropolis(X,Y,tau,d, burnin2,N2,thin2);
%save('long_run','long_run')

load('long_run')


figure(1)
clf

ids = [4] % 1:size(X,2)
for i = 1:length(ids); %1:size(X,2)
id = ids(i);
%figure(id); clf ; hold on
subplot(1,length(ids),i);  hold on;     
    
grid_pts = 150;
    ext = 0; % extra beyond points
    MIN = min(samples(:,id));
    MAX = max(samples(:,id));
    
    s = linspace(MIN-ext*abs(MIN) , ...
        MAX + ext*abs(MAX), grid_pts);
  
    
    [ff,xx] = ksdensity(samples(:,id),s);
    plot(xx,ff,'b', 'linewidth',3.5)
    
    
    [ff2,xx2] = ksdensity(long_run(:,id),s);
    plot(xx2,ff2,'k:', 'linewidth',3)
       
    est = zeros(grid_pts,1);
    st = zeros(grid_pts,1);
    
    p_prop = 0.1;
    p_samples = ceil(p_prop * size(samples,1)); % number of samples for p^\star
    
    use_all = false; 
    
    % translation trick to lower variance will be used if abs(s) < thresh
    thresh = 0;
    
    for j = 1:grid_pts
        if s(j) < thresh && s(j) > 0
            a = 1;
        elseif s(j) > -thresh && s(j) < 0 
            a= -1 ;
        else 
            a = 0;
        end
        
        L_ind = samples(:,id) < (s(j));
        R_ind = samples(:,id) > (s(j));
         
        tt = ((samples(:,id) + a) .* gr(id,:)' + 1);
        
        est1 = L_ind .* tt/(s(j) + a);
        est2 = -R_ind .* tt/(s(j) + a);
       
        use_all = false;
        
        % estimate optimal p 
        if ~use_all
             Sigma = cov(est1(1:p_samples), est2(1:p_samples));
        else 
             p_samples = 0;    
             Sigma = cov(est1, est2);
        end
         
        p_star = (Sigma(2,2) - Sigma(1,2))...
                    /(Sigma(1,1) + Sigma(2,2) - 2 * Sigma(1,2));
                
        est(j) = p_star * mean(est1((p_samples+1):end)) ...
                    + (1-p_star) * mean(est2((p_samples+1):end));
                
        st(j) = std(p_star * est1((p_samples+1):end) ...
                    + (1-p_star) * est2((p_samples+1):end));  
    end
    
     plot(s, est, 'r', 'linewidth',3.5) 
     xlabel('x', 'interpreter', 'latex', 'fontsize',22)
     ylabel('\\hatf(\v x)', 'interpreter', 'latex','fontsize',22)
     
     if i == length(id)
     legend(['KDE - n = ', num2str(N - burnin)], ['KDE - n = ', num2str(N2 - burnin2)], ['Push-Out - n = ', num2str(N - burnin)])
     end
      plot(xx2,ff2,'k:', 'linewidth',3)
      
      figure(2)
      plot(s,st,'b', 'linewidth',2)
      
     
end

figure(1)
report = true; 

if report 
    ESS = multiESS(samples) 
       
    figure(10)
    for i = 1:d
      subplot(d,1,i)
      plot(samples(:,i))
     end

    figure(11)
    for i = 1:d
      subplot(d,1,i)
      autocorr(samples(:,i),100)
    end
end

end


function samples = Metropolis(X,Y,tau,d, burnin, N, thin)
theta = zeros(1,d);
samples = zeros(N,d);
    
    for i=1:N
       for j = 1:thin
       x = mvnrnd(theta, 0.0075*eye(d)); 
       alpha = min(1, exp(LogPosterior(x,X,Y,tau)-LogPosterior(theta,X,Y,tau)));
       if(rand<=alpha)
          theta = x; 
       end
        end
       samples(i,:) = theta;
    end
    
    % get gradients of log posterior   
    samples = samples((burnin+1):end, :);
end

function val = LogPosterior(theta,X,y,tau)
% log of the posterior 
f = -(tau/2)*sum(theta.^2) ;
term  = X*theta';
L =  -sum(log(1+exp(term))) + sum(y.*term);
val = f + L;

end


function gr = gradLogPosterior(theta,X,y,tau)
% gradient of log posterior
term  = X*theta';
gr_prior = (-tau * theta)';
p = logsig(term);
gr_loglike = X' * (y-p);
gr = gr_prior + gr_loglike;
end
