
#finds the dot product of 2 2d matrices
def dot_product(x: list, y: list):

    a = len(x)
    b = len(x[0])
    c = len(y[0])

    #collect dimensions of the matrices
    # x has dims of a*b
    # y has dims of b*c

    final_matrix = [[0]*c for _ in range(a)]

    for row in range(a):
        for column in range(c):
            total = 0

            for i in range(b):
            
                total += x[row][i] * y[i][column]
                

            final_matrix[row][column] = total


    return final_matrix

A = [
    [1, 0, 2, 3],
    [4, 1, 0, 1],
    [2, 3, 1, 0]
]

B = [
    [2, 1, 0],
    [0, 3, 1],
    [1, 0, 2],
    [1, 2, 1]
]

#passes test case

print(dot_product(A,B))


#swaps the rows and columns of a 2d matrix
def transposition(x:list):
    a = len(x)
    b = len(x[0])

    final_matrix = [[0]*a for _ in range(b)]

    for column in range(a):
        for row in range(b):
            final_matrix[row][column] = x[column][row]

    return final_matrix

C = [
    [5, 6],
    [7, 8],
    [9, 10],
]

print (transposition(C))

