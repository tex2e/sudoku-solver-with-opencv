
import z3 # pip install z3-solver

from itertools import product

grid = lambda i, j: z3.Int("grid[%d,%d]" % (i, j))


class Z3Solver(z3.Solver):
    GRID_SIZE = 9
    SUB_GRID_SIZE = 3

    def __init__(self, problem):
        super(Z3Solver, self).__init__()
        self.problem = problem
        self._set_problem()

    def solve(self):
        self._set_restriction()
        return self.check()

    def _set_problem(self):
        N = Z3Solver.GRID_SIZE
        for i, j in product(range(N), range(N)):
            if self.problem[i][j] > 0:
                self.add(grid(i, j) == self.problem[i][j])

    def _set_restriction(self):
        N = Z3Solver.GRID_SIZE
        B = Z3Solver.GRID_SIZE//Z3Solver.SUB_GRID_SIZE
        SUB_GRID_SIZE = Z3Solver.SUB_GRID_SIZE
        # set initial value
        self.add(*[1 <= grid(i, j) for i, j in product(range(N), repeat=2)])
        self.add(*[grid(i, j) <= 9 for i, j in product(range(N), repeat=2)])
        # distinct w.r.t col
        for row in range(N):
            self.add(z3.Distinct([grid(row, col) for col in range(N)]))
        # distinct w.r.t row
        for col in range(N):
            self.add(z3.Distinct([grid(row, col) for row in range(N)]))
        # distinct
        for row in range(B):
            for col in range(B):
                block = [grid(B*row+i, B*col+j)
                         for i, j in product(range(SUB_GRID_SIZE), repeat=2)]
                self.add(z3.Distinct(block))


def solve_sudoku(problem):
    solver = Z3Solver(problem)
    result = solver.solve()
    if result == z3.sat:
        model = solver.model()
        # print result
        for i in range(Z3Solver.GRID_SIZE):
            for j in range(Z3Solver.GRID_SIZE):
                print("%d " % model[grid(i, j)].as_long(), end="")
            print("")
        print("")
    else:
        print(result)


def main():
    # problem1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 1, 0, 8, 0],
    #             [6, 4, 0, 0, 0, 0, 7, 0, 0],
    #             [0, 0, 0, 0, 0, 3, 0, 0, 0],
    #             [0, 0, 1, 8, 0, 5, 0, 0, 0],
    #             [9, 0, 0, 0, 0, 0, 4, 0, 2],
    #             [0, 0, 0, 0, 0, 9, 3, 5, 0],
    #             [7, 0, 0, 0, 6, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 2, 0, 0, 0, 0]]
    # solve_sudoku(problem1)
    #
    # problem2 = [[0, 0, 0, 0, 0, 0, 0, 3, 9],
    #             [0, 0, 0, 0, 0, 1, 0, 0, 5],
    #             [0, 0, 3, 0, 5, 0, 8, 0, 0],
    #             [0, 0, 8, 0, 9, 0, 0, 0, 6],
    #             [0, 7, 0, 0, 0, 2, 0, 0, 0],
    #             [1, 0, 0, 4, 0, 0, 0, 0, 0],
    #             [0, 0, 9, 0, 8, 0, 0, 5, 0],
    #             [0, 2, 0, 0, 0, 0, 6, 0, 0],
    #             [4, 0, 0, 7, 0, 0, 0, 0, 0]]
    # solve_sudoku(problem2)

    # problem3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    problem3 = [[0, 0, 5, 6, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 4, 0, 0, 5, 0],
                [0, 0, 8, 2, 0, 0, 4, 0, 6],
                [0, 0, 0, 0, 0, 0, 3, 0, 1],
                [0, 4, 0, 0, 0, 0, 0, 9, 0],
                [8, 0, 2, 0, 0, 0, 0, 0, 0],
                [2, 0, 6, 0, 0, 1, 7, 0, 0],
                [0, 9, 0, 0, 7, 0, 0, 4, 0],
                [0, 0, 0, 0, 0, 5, 8, 0, 0]]
    solve_sudoku(problem3)

if __name__ == '__main__':
    main()
