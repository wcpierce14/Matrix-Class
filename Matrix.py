"""
Matrix.py
Name(s): William "Cason" Pierce
NetId(s): wcp17
Date: 2-10-22
"""

"""
printMat

This function will print a matrix in a readable format. You will not need to
alter this function.

INPUTS
mat: the matrix represented as a list of lists in row major form.

OUTPUTS
s: a string representing the matrix formatted nicely.
"""
def printMat(mat):
    s = ''
    for row in mat:
        s = s + ''
        for col in row:
            # Display 2 digits after the decimal, using 9 chars.
            s = s + ('%9.2e' % col) + ' '
        s = s + '\n'
    return s

"""
Matrix Class

This class will have code that implements:
- matrix multiplication
- LU factorization
- backward substitution
- forward substitution
- permutation
- Gaussian elimination
"""
class Matrix:


    """
    Class attributes:
    mat:     the matrix itself, represented as a list of lists.
    numRows: the number of rows in the matrix.
    numCols: the number of columns in the matrix.
    L:       the lower triangular matrix from the LU Factorization.
    U:       the upper triangular matrix from the LU Factorization.
    ipiv:    the permutation vector from the LU Factorization.
    """

    # Constructor method.
    def __init__(self, mat):
        self.mat = mat
        self.numRows = len(mat)
        self.numCols = len(mat[0])
        self.L = None
        self.U = None
        self.ipiv = None

    # Special method used for printing this Matrix.
    # You will not have to alter this function.
    def __repr__(self):
        s = ''
        s += 'The %dx%d Matrix itself:\n\n' % (self.numRows, self.numCols)
        s += printMat(self.mat)
        s += '\n'
        if self.L != None:
            s += 'The lower triangular matrix L:\n\n'
            s += printMat(self.L.mat)
            s += '\n'
        if self.U != None:
            s += 'The upper triangular matrix U:\n\n'
            s += printMat(self.U.mat)
            s += '\n'
        if self.ipiv != None:
            s += 'The permutation vector ipiv:\n\n'
            s += printMat([self.ipiv])
            s += '\n'
        return s

    """
    This matMult method of the Matrix class computes the product of two matrices

    INPUTS
    B: a Matrix object that is multiplied with the Matrix object upon which the
    method is called

    OUTPUTS
    C: a Matrix object where the corresponding matrix of C is the product of the Matrix
    that the method is called on and B.
    """

    def matMult(self,B):
        # Throw an error if the dimensions do not allow for matrix multiplication
        if self.numRows != B.numRows:
            raise ValueError('These matrices cannot be multiplied')

        # Create an empty temporary matrix
        C = Matrix([[0.0 for x in range(B.numCols)] for y in range(self.numRows)])

        # Loop through the rows of A, the columns of B, and the rows of B to
        # Calculate the dot product of each row of A with each column of B
        for i in range(self.numRows):
            for k in range(B.numCols):
                for p in range(self.numCols):
                    # Assign the dot product to its corresponding location in temp
                    C.mat[i][k] += self.mat[i][p] * B.mat[p][k]
        return C


    """
    This method LUfact performs the LU factorization on a Matrix object and asssigns the 
    matrices L, U, and the vector ipiv to the corresponding attributes of the Matrix 
    object in which the method is called on.
    
    INPUTS
    There are no explicit parameters for this method, however the method must be called on
    an Object of the Matrix class
    
    OUTPUTS
    There is no return value
    """
    def LUfact(self):
        # Create a copy of self.mat so that the original matrix is left unaltered
        U = [self.mat[ind].copy() for ind in range(self.numRows)]

        # Create a permutation vector ipiv that begins as a vector corresponding
        # to the rows of A and changes as operations are done on U
        ipiv = [i for i in range(self.numRows)]

        # Create the matrix L that begins as the identity matrix and is
        # updated with elimination steps and row swaps
        L = [[0.0] * j + [1] + [0.0] * (self.numRows - 1 - j) for j in range(self.numRows)]

        # For each row of the matrix (excluding the last)...
        # Recall that the last row is at index n-1.
        for j in range(self.numRows - 1):
            # Select the maximal pivot.
            pivotRow = j

            for i in range(j, self.numRows):
                # If the new potential pivot is larger, update.
                if abs(U[i][j]) > abs(U[pivotRow][j]):
                    pivotRow = i

            # If we cannot find a non-zero pivot, raise an error because the
            # matrix is singular
                if U[pivotRow][j] == 0:
                    raise ValueError('This matrix is singular')

            # When performing the corresponding row swaps for L,
            # only swap the elements below the diagonal
            for h in range(j):
                tempElement = L[j][h]
                L[j][h] = L[pivotRow][h]
                L[pivotRow][h] = tempElement

            # Swap the rows for U
            tempRow = U[j]
            U[j] = U[pivotRow]
            U[pivotRow] = tempRow

            # Swap the rows for ipiv
            tempElement = ipiv[j]
            ipiv[j] = ipiv[pivotRow]
            ipiv[pivotRow] = tempElement

            # Eliminate elements in U by canceling them using row operations
            # Populate L with the corresponding elimination operations
            for k in range(j + 1, self.numRows):
                l = U[k][j] / U[j][j]
                L[k][j] = l
                for c in range(j, self.numRows):
                    U[k][c] += -l * U[j][c]


        # If the element in the bottom right corner of matrix U is 0,
        # the matrix is singular
        if U[len(U) - 1][len(U) - 1] == 0:
            raise ValueError('This matrix is singular')

        # Assign the L, U, and ipiv attributes to the L, U, and ipiv
        # matrices and vector calculated from this LU decomposition
        self.L = Matrix(L)
        self.U = Matrix(U)
        self.ipiv = ipiv

        return

    """
    This method backSub performs backwards substitution on an upper triangular
    Matrix object to solve the equation Ux = c
    
    INPUTS
    c: The resulting Matrix object from performing the forward elimination's 
    elementary row operations on b.
    
    OUTPUTS
    x: The Matrix object that is the solution to the equation Ux = c where
    U is the Matrix upon which the method is called and c is the input
    """
    def backSub(self,c):
        # Throw an error if the row dimension of U does not match the
        # row dimension of c
        if self.numRows != c.numRows:
            raise ValueError('Backwards Substitution cannot be computed')

        # Set n as the number of rows in U
        n = self.numRows

        # Create an empty matrix for x with the correct dimensions
        x = [[0.0] for i in range(n)]

        # Perform the first calculation to get the value corresponding to
        # the last row in x
        x[n - 1][0] = c.mat[n - 1][0] / self.mat[n - 1][n - 1]

        # Loop backwards over the rows of U
        for i in range(n - 1):
            j = n - 2 - i
            sum = 0

            # Loop over the elements to the right of the diagonal to calculate the
            # sum seen below
            for k in range(j + 1, n):
                sum += self.mat[j][k] * x[k][0]

            # Populate the corresponding values of x
            x[j][0] = 1 / self.mat[j][j] * (c.mat[j][0] - sum)

        # Convert x into a Matrix object
        x = Matrix(x)
        return x


    """
    This method forSub performs forwards substitution on a lower triangular 
    Matrix object to solve the equation Lc = b
    
    INPUTS
    b: the Matrix object corresponding to the product of the original equation Ax = b
    
    OUTPUTS
    c: the Matrix object that is the solution to the equation Lc = b where L is a lower
    triangular matrix
    """
    def forSub(self,b):

        # Throw an error if the row dimension of L does not match the
        # row dimension of b
        if self.numRows != b.numRows:
            raise ValueError('Forwards Substitution cannot be computed')

        # Let n correspond to the number of rows in L
        n = self.numRows

        # Let c be the empty solution vector with the proper dimensions
        c = [[0] for i in range(n)]

        # Set the first row of c equal to the first row of b
        c[0][0] = b.mat[0][0]

        # Loop over the subsequent rows of L
        for j in range(1, n):
            sum = 0
            # Compute each value of the vector c
            for k in range(n):
                sum += self.mat[j][k] * c[k][0]
            c[j][0] = b.mat[j][0] - sum

        # Convert c into a Matrix object
        c = Matrix(c)

        return c


    """
    This method permute rearranges the elements from Matrix object b corresponding
    to the row operations from the self.ipiv where self is the Matrix object upon
    which the method is called
    
    INPUTS
    b: the Matrix object corresponding to the product of the original equation Ax = b
    
    OUTPUTS
    bHat: the Matrix object which has the elements from b in the correct order corresponding
    to the row operations made during the LU factorization
    """
    def permute(self,b):
        # Throw an error if the row dimension of b does not match the
        # row dimension of ipiv
        if b.numRows != self.numRows:
            raise ValueError('We cannot permute b using ipiv')

        # Populate bHat by reordering the elements of b based on the
        # order of elements in ipiv
        bHat = [b.mat[j] for j in self.ipiv]

        # Convert bHat into a Matrix object
        bHat = Matrix(bHat)

        return bHat


    """
    This method gaussElim performs Gaussian elimination to solve the matrix equation Ax = b
    where A is the Matrix object upon which the method is called
    
    INPUTS
    b: the Matrix object corresponding to the product of the original equation Ax = b
    
    OUTPUTS
    x: the Matrix object corresponding to the solution of the original equation Ax = b
    """
    def gaussElim(self,b):
        # Throw an error if the row dimension of A does not match the
        # row dimension of b
        if self.numCols != b.numRows:
            raise ValueError('We cannot perform Gaussian Elimination to solve Ax = b')

        # Set self.L, self.U, and self.ipiv to the resulting outputs of the LU factorization function
        # if they have not already been computed
        if (self.L == None) and (self.U == None) and (self.ipiv == None):
            self.LUfact()

        # Compute bHat by permuting the rows of b
        bHat = self.permute(b)

        # Calculate c by performing forwards substitution
        c = self.L.forSub(bHat)

        # Calculate x by performing backwards substitution
        x = self.U.backSub(c)

        return x


    """
    This method overrides the * operator by using the matMult method to compute
    the product of two Matrix objects
    """
    def __mul__(self, B):
        return self.matMult(B)


    if __name__ == "__main__":
        from Matrix import Matrix
        table = [[1,2,3,4], [4,5,6,-19], [7,8,9,41], [1, 78, -98, 5]]
        product = [[1], [2], [3], [4]]
        A = Matrix(table)
        b = Matrix(product)
        print(A.gaussElim(b))
