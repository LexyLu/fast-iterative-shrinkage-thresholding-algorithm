# fast-iterative-shrinkage-thresholding-algorithm

## Reference
Fast Iterative Shrinkage-Thresholding Alogorithm for Linear Inverse Problems
- https://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf

## Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) preserves the computational simplicity of [ISTA](https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning) but with a global rate of convergence which is proven to be significantly better, both theoretically and practically.

## Cost function 
Cost function is fomulated by data fidelty term `1/2 * || A(x) - y ||_2^2` and l1 regularization term `L * || X ||_1` as follow,

        (P1) arg min_x [ 1/2 * || A(x) - y ||_2^2 + L * || x ||_1 ].

Equivalently,

        (P2) arg min_x [ 1/2 * || x - x_(k) ||_2^2 + L * || x ||_1 ],

where,

        x_(k) = x_(k-1) - t_(k) * AT(A(x) - y) and t_(k) is step size. 

(P2) is equal to `l1` [proximal operator](https://en.wikipedia.org/wiki/Proximal_operator),

        (P2) = prox_(L * l1)( x_(k) )

             = soft_threshold(x_(k), L)

where, 

        l1 is || x ||_1 and L is thresholding value.

`soft_threshold` is defined by,

        function y = soft_threshold(x, L)
            y = (max(abs(x) - L, 0)).*sign(x);
        end

## The basic iteration FISTA for solving problem (P1 = P2)
        for k = 1 : N
            x_(k)   = prox_(L * l1)( y_(k) )
            
            t_(k+1) = ( 1 + sqrt( 1 + 4*t_(k)^2 ) )/2
            
            y_(k+1) = x_(k) + ( t_(k) - 1 )/( (t_(k+1) ) * ( x_(k) - x_(k-1) ) 
        end

where, 

        prox_(L * l1)( x ) = soft_threshold(x, L).
