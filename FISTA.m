%% REFERENCE
% https://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf

%% 
% COST FUNCTION
% x^* = argmin_x { 1/2 * || A(X) - Y ||_2^2 + lambda/2 * ( || X ||_1 ) }
%
% x^k+1 = threshold(x^k - 1/L*AT(A(x^k)) - Y), lambda/L)

%%
function [x, obj]  = FISTA(A, AT, x, b, LAMBDA, L, n, COST, bfig)

if (nargin < 9)
    bfig = false;
end

if (nargin < 8)
    COST.function	= @(x) (0);
    COST.equation	= [];
end

if (nargin < 7)
    n   = 1e2;
end

y       = x;
xpre    = x;
T       = 1;
obj     = zeros(n, 1);

for i = 1:n
    
    x       = threshold(y - 1/L*AT(A(y) - b), LAMBDA/L);
    
    Tpost   = (1 + sqrt(1 + 4*T^2))/2;
    y       = x + (T - 1)/Tpost * (x - xpre);
    
    xpre    = x;
    T       = Tpost;
    
    obj(i)  = COST.function(x);
    
    if bfig
        figure(1); colormap gray;
        subplot(121); imagesc(x);           title([num2str(i) ' / ' num2str(n)]);
        subplot(122); semilogy(obj, '*-');  title(COST.equation);  xlabel('# of iteration');   ylabel('Objective');
        xlim([1, n]);   grid on; grid minor;
        drawnow();
    end
end

x   = gather(x);
obj = gather(obj);

end
