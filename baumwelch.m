function [A_estimate, E_estimate] = baumwelch(data, A_guess, E_guess, N_iter)
%
% Train Hidden Markov Model using the Baum-Welch algorithm (expectation maximization)
% Input:
%  data: N*T matrix, N data samples of length T
%  A_guess: K*K matrix, where K is the number hidden states [initial guess for the transition matrix]
%  E_guess: K*E matrix, where E is the number of emissions [initial guess for the emission matrix]
%
% Output:
%  A_estimate: estimate for the transition matrix after N_iter iterations of expectation-maximization
%  E_estimate: estimate for the emission matrix after N_iter iterations of expectation-maximization
%
% CSCI 576 2014 Fall, Homework 5
new_data=[zeros(length(data(:,1)),1),data];
[N,T]=size(new_data);
[K,E]=size(E_guess);

alpha=zeros(K,T);
beta=zeros(K,T);
gamma=zeros(K,T);
xi_T=zeros(K,K);
xi=zeros(K,K);

A=A_guess;
B=E_guess;

%Pi=repmat(1/K,K,1);
Pi=[0.2,0.8];

X=new_data;
%B_T=zeros(K,T);

for iter=1:N_iter
    A_new=zeros(K,K);
    B_new=zeros(K,E);
    for n=1:N
        %%% Forward
        
        alpha=zeros(K,T);
        
        scale=zeros(1,T);
        alpha(1,1)=1;
        %alpha(:,1)=Pi(:).*B(:,X(n,1));
        
        for t=2:T
            for i=1:K
                buffer=0;
                for j=1:K
                    buffer=buffer+alpha(j,t-1)*A(j,i);
                end
                alpha(i,t)=buffer*B(i,X(n,t));
            end
            scale(t) =  sum(alpha(:,t));
            alpha(:,t) =  alpha(:,t)./scale(t);
        end
        
        alpha(isnan(alpha)) = 0;
        
        %%% Backward
        
        
        beta= ones(K,T);
        
        for h=1:T-1
            t=T-h;
            for i=1:K
                beta(i,t) = (1/scale(t+1)) * sum( A(i,:)'.* beta(:,t+1) .* B(:,X(n,t+1)));
                
            end
        end
        
        beta(isnan(beta)) = 0;
        %%%% B
        
        for i=1:K
            for e=1:E
                pos = find(X(n,:) == e);
                %for p=1:length(pos)
                B_new(i,e)=B_new(i,e)+sum(alpha(i,pos).*beta(i,pos));
                %end
            end
        end
        
        B(isnan(B)) = 0;
        %%% A
        
        for i=1:K
            for j=1:K
                for t=1:T-1
                    
                    A_new(i,j)=A_new(i,j) + (alpha(i,t)*A(i,j)*B(j,X(n,t+1))*beta(j,t+1))./scale(t+1);
                end
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        A_new(isnan(A_new)) = 0;
        
    end
    
    
    sum_A=sum(A_new,2);
    sum_B=sum(B_new,2);
    
    A=A_new./repmat(sum_A,1,K);
    B=B_new./repmat(sum_B,1,E);
    
    
    %%% pi
    Pi(:)=gamma(:,1);
    
end


A_estimate=A;
E_estimate=B;
