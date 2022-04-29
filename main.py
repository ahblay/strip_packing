import csv, os
import numpy as np
from pprint import pprint as pp
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, LpBinary, GUROBI_CMD


# read in input data
# each line contains three values corresponding to the width, height and quantity of rectangles to pack onto a larger
# rectangle of minimum height and fixed width
def read_data(filename):
    # data is a list of tuples
    data = []

    with open(filename, newline='') as f:
        r = csv.reader(f)
        for row in r:
            t = [int(i) for i in row]
            data.append(tuple(t))

    return data


# a class that handles the linear optimization problem
class TwoSP:

    # initialize the class with:
    # the set of rectangles to be packed
    # the width and initial height of strip these rectangles are to be packed onto
    # the LP model
    # the correspondence matrix, as defined in https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0245267
    # the set of variables
    # the constraints and objective function
    def __init__(self, data, w, h):
        self.data = data
        self.w = w
        self.h = h
        self.model = LpProblem(name='2SP', sense=LpMinimize)
        self.cm, self.cd = self.get_correspondence_matrix()
        self.variable_matrix = [[LpVariable(f'x_{i}.{j}', cat=LpBinary)
                                 for j in range(len(self.cd[i]))]
                                for i in self.cd.keys()]
        self.variable_list = [v
                              for i in range(len(data))
                              for v in self.variable_matrix[i]]
        self.build_constraints()
        self.build_objective()
        self.output = []

    # calculates the lower bound on the height of the strip, as a function of the total area of rectangles to
    # be packed
    @staticmethod
    def get_lower_bound(data, w):
        h = 0
        for item in data:
            h += item[0] * item[1] * item[2]
        return int(h/w)

    # a helper function for building the correspondence matrix, as described in:
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0245267
    @staticmethod
    def find_ones(rect, w, h, coord):
        result = []
        if coord[0] + rect[0] <= w and coord[1] + rect[1] <= h:
            ones = []
            for row in range(coord[1], coord[1] + rect[1]):
                ones += list(range(row * w + coord[0], row * w + coord[0] + rect[0]))
            for i in range(w * h):
                if i in ones:
                    result.append(1)
                else:
                    result.append(0)
        return result

    @staticmethod
    def get_xy(idx, w):
        x = idx % w
        y = int(idx / w)
        return x, y

    def get_box_index(self, idx, rect):
        for box_index, val in enumerate(self.cd[rect][idx]):
            if val == 1:
                return box_index

    def get_correspondence_matrix(self):
        c_matrix = []
        c_dict = {}

        for item in self.data:
            c_dict[item] = []
            for j in range(self.h):
                for i in range(self.w):
                    row = self.find_ones(item, self.w, self.h, (i, j))
                    if row:
                        c_dict[item].append(row)
                        for _ in range(item[2]):
                            c_matrix.append(row)

        return np.array(c_matrix), c_dict

    # builds the constraints
    def build_constraints(self):

        # prevents tiles from overlapping
        for col in range(self.w * self.h):
            self.model += (lpSum(self.cm[index][col] * self.variable_list[index]
                                 for index in range(len(self.variable_list))) <= 1,
                           f'constraint_3.{col}')

        # ensures that all rectangles are packed onto the strip
        for rect in range(len(self.data)):
            self.model += (lpSum(self.variable_matrix[rect][location]
                                 for location in range(len(self.cd[self.data[rect]]))) >= self.data[rect][2],
                           f'constraint_4.{rect}')
        
        # ensures that no additional rectangles are packed onto the strip
        for rect in range(len(self.data)):
            self.model += (lpSum(self.variable_matrix[rect][location]
                                 for location in range(len(self.cd[self.data[rect]]))) <= self.data[rect][2],
                           f'constraint_5.{rect}')

        # prevents the areas of the rectangles from exceeding that of the strip
        self.model += (lpSum(self.cm[index][col] * self.variable_list[index]
                             for col in range(self.w * self.h)
                             for index in range(len(self.variable_list))) <= self.w * self.h,
                       'constraint_6')
        return

    # adds the objective function
    # in this case, we simply wish to satisfy all the constraints for the lowest possible value of h,
    # so the objective function is held constant
    def build_objective(self):
        self.model += 0
        return

    # run the gurobi solver
    def solve(self):
        self.model.solve(GUROBI_CMD())
        print(self.model.status)
        return self.model.status

    # as each variable corresponds to a position for the rectangles, we find the coordinates of those variables
    # that are set equal to 1
    # this information is used to draw the resulting figure in latex
    def get_output(self):
        output = []
        for rect_index, rect in enumerate(self.variable_matrix):
            for location_index, location in enumerate(rect):
                if location.value() == 1.0:
                    box_index = self.get_box_index(location_index, data[rect_index])
                    item = [self.data[rect_index], (self.get_xy(box_index, self.w))]
                    output.append(item)
        self.output = output
        return output

    # render the output data into a latex figure
    def to_latex(self):

        # all of the basic code for rendering a LaTeX file
        header = [
            '\documentclass{article}',
            '\\usepackage[margin=1in]{geometry}',
            '\\usepackage{tikz}',
            '\setlength\parindent{0pt}',
            '\\renewcommand{\\familydefault}{\\ttdefault}',
            '\\begin{document}',
            '\\begin{tikzpicture}'
        ]
        footer = [
            '\end{tikzpicture}',
            '\end{document}'
        ]

        # a somewhat hacky solution to construct a LaTeX document where each line has the appropriate spacing
        with open('output.tex', 'w') as f:
            f.writelines('\n'.join(header))
            f.writelines('\n')
            f.writelines(f'\draw[draw=black] (0,0) rectangle ({self.w},{self.h});')
            for item in self.output:
                f.writelines('\n')
                f.writelines(f'\\filldraw[draw=black, fill=red] ({item[1][0]},{item[1][1]}) rectangle '
                             f'({item[1][0] + item[0][0]},{item[1][1] + item[0][1]});')
            f.writelines('\n')
            f.writelines('\n'.join(footer))

        # compiles the document
        os.system("pdflatex output.tex")

# read in the data
data = read_data('input.csv')
print(data)

# set the width of the strip
w = 8
# calculate the lower bound on th height of the strip and initialize the problem
h = TwoSP.get_lower_bound(data, w)
problem = TwoSP(data, w, h)
print(h)

# while the problem remains unsolved (infeasible), increment the height and attempt to solve again
while problem.solve() == 0:
    h += 1
    print(h)
    problem = TwoSP(data, w, h)

for var in problem.model.variables():
    print(f"{var.name}: {var.value()}")

# print the resulting solution and render it in latex
output = problem.get_output()
print(output)
problem.to_latex()