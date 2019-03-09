# import numpy as np
# import scipy.optimize
# from random import uniform as u
from math import exp, cos, sin, pi


def main():

    def kissSophie(howOften=3):
        for time in range(howOften):
            print("Kiss :*")
        return "lol"

    a = kissSophie(100)
    print(a)


def norm(array):
    """ return the Euclidian Norm of a list"""
    return (sum([i**2 for i in array]))**0.5


def dot(array1, array2):
    """ simple dot product"""
    return sum([array1[i] * array2[i] for i in range(len(array1))])


class best_circle(object):
    """Finds the best fitting circle """

    def __init__(self, points):
        super(best_circle, self).__init__()

        # setting parameters
        self.POINTS = [np.array([point[0], point[1]]) for point in points]

        # Inital guess for the midpoint should be the center of mass
        INITIAL_GUESS_MIDPOINT = np.mean(
            self.POINTS, axis=0) + np.array([0.001, 0.001])

        # initial radius is the mean deviance from the center of mass
        deviance_from_mean = 0
        for point in self.POINTS:
            deviance_from_mean += norm(point - INITIAL_GUESS_MIDPOINT)

        INITIAL_GUESS_RADIUS = deviance_from_mean / len(self.POINTS)
        self.INITIAL_GUESS = np.hstack(
            [INITIAL_GUESS_RADIUS, INITIAL_GUESS_MIDPOINT])

        # setting initial guess for iterative procedure
        self.best_guess = np.hstack(
            [INITIAL_GUESS_RADIUS, INITIAL_GUESS_MIDPOINT])

        # finding the best cirle
        self.fitted_circle = self.INITIAL_GUESS

    def deviance(self, point):
        """ Error contribution from each point given the current best guess"""
        x, y = point
        r, mid_x, mid_y = self.best_guess

        return r - ((x - mid_x)**2 + (y - mid_y)**2)**0.5

    def objective_function(self):
        """ Function that the optimal circle should minimise.
                Given by the sum of the deviances for each point"""
        objective_value = 0
        for point in self.POINTS:
            objective_value += self.deviance(point)**2

        return objective_value

    def gradient(self):
        """ Gradient of the objective function at the current guess"""
        dm_x = 0
        dm_y = 0
        dr = 0
        r, mx, my = self.best_guess

        for point in self.POINTS:
            D = self.deviance(point)
            x, y = point
            dx = x - mx
            dy = y - my
            n = (dx**2 + dy**2)**0.5

            dr += r * D
            dm_x += dx * D / n
            dm_y += dy * D / n

        gradient = 2 * np.array([dr, dm_x, dm_y])

        return np.ndarray.flatten(gradient)

    def minimise_objective_function_BFGS(self):
        """ Running the BFGS minimisation """
        result = scipy.optimize.minimize(fun=self.objective_function,
                                         jac=self.gradient,
                                         method="BFGS")
        self.best_guess = result.x

    def hessian_part(self, point):
        """ gets the hessian contribution from each point"""
        x, y = point
        r, mx, my = self.best_guess

        dx = (x - mx)
        dy = (y - my)
        n = (dx**2 + dy**2 + 0.0001)**0.5

        # constructing diagonal elements
        H11 = 1
        H22 = 1 - r / n + r * dx**2 / n**3
        H33 = 2 - r / n + r * dy**2 / n**3

        diagonal = np.diag(np.array([H11, H22, H33]))

        # upper triangle
        H12, H13, H23 = dx / n, dy / n, r * dx * dy / n

        H = np.zeros((3, 3))
        H[0, 1], H[0, 2], H[1, 2] = H12, H13, H23
        Ht = np.transpose(H)
        H = H + Ht + diagonal
        return H

    def hessian(self):
        """ returns the Hessian """
        H = np.zeros((3, 3))
        for point in self.POINTS:
            H += self.hessian_part(point)
        return 2 * H

    def Newton(self, max_iterations=50, tol=0.1):
        """Ordinary Newton's method"""

        for iteration in range(max_iterations):
            # main iterations

            # Finding current Hessian an Gradient
            H = self.hessian()
            F = -1 * self.gradient()

            # Uncomment for debugging
            # print("Hessian: \n" + str(H))
            # print("-Grad: \n" + str(F))

            # Solve for next step
            zn = np.linalg.solve(H, F)

            # Go next step
            self.best_guess += zn
            print(self.best_guess)

            # Stop if the next step is tiny
            if norm(zn) < tol:
                self.fitted_circle = self.best_guess
                print("Converged!")
                return self.best_guess

            # random restart should the algorithm
            # seem to diverge
            elif norm(zn) > 1000 or self.best_guess[0] < -1:
                self.best_guess = np.array([u(i / 2, 3 / 2 * i)
                                            for i in self.INITIAL_GUESS])
                print("Diverged, resetting!")
                print("New Guess:" + str(self.best_guess))
        print("Did not converge!")

    # methodes for evaluating goodness of fit
    def score(self):
        # firstly, we center the data around the center
        # of the best fit circle
        POINTS = self.POINTS - self.fitted_circle[1:2]

        # now we scale them down such that the best fitted
        # circle has radius 1
        POINTS = POINTS / self.fitted_circle[0]

        # the first quantity of interest is
        # d_mass: deviance of the center of mass
        center_of_mass = np.mean(POINTS, axis=0)
        d_mass = norm(center_of_mass)

        # the next quanitity is the average distance
        # from the circumference d_radius
        total_deviance = 0
        for point in POINTS:
            total_deviance += abs(norm(point) - 1)
        d_radius = total_deviance / len(POINTS)

        # lastly, we find the angle to some base vector
        # and check if the angles are not too far apart
        base_vector = np.array([0, 1])
        angles = []

        for point in POINTS:
            point = point / norm(point)
            angles.append(dot(point, base_vector) / 2)

        # sort angles to find biggest gap
        angles = sorted(angles)
        d_angle = 0
        for index in range(len(angles) - 1):
            gap = abs(angles[index] - angles[index + 1])
            if gap > d_angle:
                d_angle = gap

        # return total score (number between 0 (worst) and 1 (best))
        self.score = (1 - d_angle) * exp(-(d_mass + d_radius))
        print("score:" + str(self.score))
        return self.score


if __name__ == '__main__':
    main()
