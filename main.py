import numpy as np
from matplotlib import pyplot as plt 


'''
Implement a function that given the domain interval, the forcing function, the number of discretization points, 
the boundary conditions, returns the matrix  and the the right hand side b.
'''
def finDif(omega, f, n, bc):
    span = omega[ -1 ] - omega[ 0 ]
    delta = span / ( n - 1 )

    # A's diagonals 
    diags = [
        30 * np.ones( (n,) ),
        -16 * np.ones( (n - 1,) ),  
        np.ones( (n - 2,) ) 
    ]

    A = np.diag( diags[ 0 ], 0 ) \
        + np.diag( diags[ 1 ], -1 ) \
        + np.diag( diags[ 1 ], 1 ) \
        + np.diag( diags[ 2 ], -2 ) \
        + np.diag( diags[ 2 ], 2 )
    
    A /= (12 * delta**2) 

    x = np.linspace( omega[ 0 ], omega[ -1 ], n )
    b = f( x )

    # boundary conditions
    A[ 0, : ] = 0
    A[ :, 0 ] = 0
    A[ 0, 0 ] = 1
    b[ 0 ] = bc[ 0 ]

    A[ -1, : ] = 0
    A[ :, -1 ] = 0
    A[ -1, -1 ] = 1
    b[ -1 ] = bc[ -1 ] 
    return (A, b)


omega = [ 0, np.pi ]
f = lambda x : np.sin(x)

n = 100
bc = [ 0, 0 ]
(A, b) = finDif( omega, f, n, bc )

print(A)
print(b)

'''
Implement two functions that compute the LU and the Cholesky factorization of the system matrix A
'''
def LU(A, tol=1e-15):
    A = A.copy()
    size = len( A )

    for k in range(size - 1):
        pivot = A[ k, k ]
        if abs(pivot) < tol:
            raise RuntimeError("Null pivot")

        for j in range( k + 1, size ): 
            A[ j, k ] /= pivot 

        for j in range( k + 1, size ):
            A[ k + 1 : size, j ] -= A[ k + 1 : size, k ] * A[ k, j ] 

    L = np.tril( A )
    for i in range( size ):
        L[ i, i ] = 1.0

    U = np.triu( A )
    return (L, U)

(L, U) = LU( A )
print(L, U)


def cholesky(A):
    A = A.copy()
    size = len( A )
    
    for k in range( size - 1 ):
        A[ k, k ] = np.sqrt(A[ k, k ])

        A[ k + 1 : size, k ] = A[ k + 1 : size, k ] / A[ k, k ]

        for j in range(k+1,size):
            A[ j : size, j ] = A[ j : size, j ] - A[ j : size, k ] * A[ j, k ]

    
    A[ -1, -1 ] = np.sqrt(A[ -1, -1 ])

    L = np.tril( A )
    Lt = L.transpose()
    return (L, Lt)

(Ht, H) = cholesky( A )
print(Ht, H)


# Implement forward and backward substitution functions to exploit the developed factorization methods to solve the 
# derived linear system of equations.


def L_solve(L, rhs):
    size = len( L )
    x = np.zeros( size )

    x[ 0 ] = rhs[ 0 ] / L[ 0, 0 ]

    for i in range( 1, size ):
        x[ i ] = ( rhs[ i ] - np.dot( L[ i, 0 : i ], x[ 0 : i ] ) ) / L[ i, i ]
    return x

def U_solve(U, rhs):
    size = len( U )
    x = np.zeros( size )

    x[ -1 ] = rhs[ -1 ] / L[ -1, -1 ]

    for i in reversed( range( size - 1 ) ):
        x[ i ] = ( rhs[ i ] - np.dot( U[ i, i + 1 : size ], x[ i + 1 : size ] ) ) / U[ i, i ]
    return x

'''
Solve the derived linear system using the implemented functions and plot the computed solution:
'''

(fig, ax) = plt.subplots()

# exact solution 
x = np.linspace( omega[0], omega[-1], n)
u_exact = np.sin( x )

ax.plot( x, u_exact, label = 'exact' )


# using LU factorization
w = L_solve( L, b )
u = U_solve( U, w )

ax.plot( x, u, label = 'lu' )


# using cholesky factorizations 
w = L_solve( Ht, b )
u = U_solve( H, w )

ax.plot( x, u, label = 'cholesky' )

ax.legend()
ax.grid()

fig.savefig( 'plot1.svg' )


'''
Considering the new domain [0, 1] and the forcing term  with B.C. , 
on  produce a plot and a table where you show the decay of the error w.r.t. the number of grid points. 
(The analytical solution for the above problems is )
'''

def compute_errors(omega, f, fex, bc, npoints):
    errors = []

    for idx in range( len(npoints) ):
        npts = npoints[ idx ]
        
        x = np.linspace( omega[0], omega[1], npts )
        ex = fex( x )

        ( A, b ) = finDif( omega, f, npts, bc )
        ( L, U ) = LU( A )
        w = L_solve( L, b )
        u = U_solve( U, w )

        err = sum( ( ex - u )**2 )**0.5
        errors.append( err )
    return errors    


omega = [ 0, 1 ]

def func(x): 
    return x * (1 - x)

def fex(x): 
    return x**4/12 - x**3/6 + x/12

bc = [ 0, 0 ]

npoints = np.arange( 10, 310, 10 )

errors = compute_errors( omega, func, fex, bc, npoints )

(fig, ax) = plt.subplots()

ax.set_yscale( 'log' )
ax.plot( npoints, errors )

fig.savefig( 'plot2.svg' )


'''
Exploit the derived LU factorizations to compute the condition number of the system's matrix  using the original problem formulation.
'''
def PM(A, z0, tol=1e-12, nmax=10000):
    q = z0 / np.linalg.norm( z0, 2 )

    it = 0
    err = tol + 1
    
    while it < nmax and err > tol:
        z = A.dot( q )
        l = q.T.dot( z )
        err = np.linalg.norm( z - l*q, 2 )
        q = z / np.linalg.norm( z, 2 )
        it = it + 1
    return (l, q)

def IPM(A, x0, mu, eps=1.0e-12, nmax=10000):
    M = A - mu * np.eye(len(A))

    (L, U) = LU( M )
    q = x0 / np.linalg.norm( x0, 2 )
    
    err = eps + 1.0
    it = 0
    while err > eps and it < nmax:
        y = L_solve( L, q )
        x = U_solve( U, y )
        q = x / np.linalg.norm( x, 2 )
        z = A.dot( q )
        l = q.T.dot( z )
        err = np.linalg.norm( z - l*q, 2 )
        it = it + 1
    return (l, q)


def condNumb(A):
    z0 = np.ones( ( len(A), ))
    lmax = PM( A, z0 )[ 0 ]
    lmin = IPM( A, z0, 0.0 )[ 0 ]

    condNum = lmax / lmin
    return condNum

condNum = condNumb( A )
print( condNum )


'''
Implement a preconditioned Conjugant Gradient method to solve the original linear system of equations using an iterative method:

'''
def conjugate_gradient(A, b, P, nmax=len(A), eps=1e-10):
    x = np.zeros_like( b )
    r = b - A.dot( x )

    rho0 = 1 
    p0 = np.zeros_like( b )

    err = eps + 1.0
    
    it = 1
    while it < nmax and err > eps:
        z = np.linalg.solve( P, r )
        rho = r.dot( z )

        if it > 1:
            beta = rho / rho0 
            p = z + beta * p0
        
        else:
            p = z

        q = A.dot( p )
        alpha = rho / p.dot( q )

        x += p * alpha
        r -= q * alpha
        
        p0 = p
        rho0 = rho

        err = np.linalg.norm( r, 2 )
        it = it + 1

    print( f'iterations: {it}' )
    print( f'error: {err}' )
    return x


omega = [ 0, np.pi ]
x = np.linspace( omega[ 0 ], omega[ -1 ], n )

ex = np.sin( x )

u = conjugate_gradient( A, b, np.diag(np.diag( A ) ))

(fig, ax) = plt.subplots()

ax.plot( x, ex, label = 'exact' )
ax.plot( x, u, label = 'CG' )

ax.legend()
fig.savefig( 'plot3.svg' )

exit( 0 )

'''
Consider the following time dependent variation of the PDE starting from the orginal problem formulation:

for , with  and 

Use the same finite difference scheme to derive the semi-discrete formulation and solve it using a forward Euler's method.

Plot the time dependent solution solution at , , 
'''

#TODO 
'''
Given the original  system, implement an algorithm to compute the eigenvalues and eigenvectors of the matrix . 
oExploit the computed LU factorization
'''

#TODO

'''
Compute the inverse of the matrix A exploiting the derived LU factorization

'''
#TODO 

'''
Consider the following Cauchy problem
 
Implement a Backward Euler's method in a suitable function and solve the resulting non-linear equation using a Newton's method.
'''

#TODO
