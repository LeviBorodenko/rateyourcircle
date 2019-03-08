"use strict"

const MAX_NUMBER_OF_POINTS = 10000;
const MAX_ITERATIONS = 50;
const TOLERANCE = 0.01;
const MAX_NEWTON_ITERATIONS = 10;

function printToPage(text) {
    /*
    Debugging class. Prints to html.
    */
    document.querySelector('div.main-col').innerHTML = text
};

function norm(array) {
    // convert to Array
    let sum = 0;
    for (let element of array) {
        sum += element ** 2
    }
    return sum ** 0.5
}

class Points {
    /*
    First step in the process of finding the best
    circle. 
    Here we test if the points are valid and define
    some attributes unique to the initialised data.
    */
    constructor(listOfPoints) {
        this.points = listOfPoints
        // setting the best guess to be the initial guess
        this.bestGuess = [...this.initialGuess]

        // setting some later used flags to default values
        this.newtonConverged = false;
        this.iterationCount = 0;
    }

    get guessMidpoint() {
        /*
        Setting the initial guess for the midpoint
        to be the (component-wise) sample mean.
        */
        let [sumX, sumY] = [0, 0]
        let n = this.points.length

        for (let point of this.points) {
            sumX += point[0]
            sumY += point[1]
        };

        return [sumX / n, sumY / n]
    }

    get guessRadius() {
        /*
        Initial guess for the radius, is the square root
        of the empirical mean square error from the sample mean
        */
        let midPoint = this.guessMidpoint;
        let radius = 0;
        for (let point of this.points) {
            radius += math.distance(midPoint, point) ** 2
        }

        return (radius / this.points.length) ** 0.5
    }

    get initialGuess() {
        /*
        Initial guess for the Newton Raphson algorithm
        */

        let [guessX, guessY] = this.guessMidpoint

        return [this.guessRadius, guessX, guessY]
    }


    get errorCode() {
        /*
        Screens the data for potential issues.

        0 - all clear
        1 - invalid data format
        2 - too few data points
        3 - too many data points
        4 - co-linear data
        5 - 
        */
        if (this.points.constructor != Array) {
            return 1
        } else if (this.points.length <= 3) {
            return 2
        } else if (this.points.length >= MAX_NUMBER_OF_POINTS) {
            return 3
        } else {
            /*
            Need to add more error codes
            */
            return 0
        }
    }
}


class Solver extends Points {
    /*
    Here we add methods related to finding the 
    best circle to the basic "points" object.
    */
    deviance(point) {
        /*
        Calculates the error contribution from
        the given point to the current best circle
        */
        let [r, midX, midY] = this.bestGuess;
        let [x, y] = point;

        return r ** 2 - ((x - midX) ** 2 + (y - midY) ** 2)
    };

    gradient() {
        /*
        Gradient of the objective function at the current
        best guess.
        */
        let [r, midX, midY] = this.bestGuess;
        let [dr, dmidX, dmidY] = [0, 0, 0];

        for (let point of this.points) {

            let D = this.deviance(point);
            let [x, y] = point;
            let [dx, dy] = [x - midX, y - midY];

            dr += r * D
            dmidX += dx * D
            dmidY += dy * D

        }

        dr *= 4
        dmidX *= 4
        dmidY *= 4

        return [dr, dmidX, dmidY]
    }

    hessianContribution(point) {
        /* 
        Contribution of point to the Hessian
        */

        // upper triangle of Hessian
        let H00, H01, H02, H11, H12, H22;

        // quantities needed
        let D = this.deviance(point)
        let [x, y] = point
        let [r, midX, midY] = this.bestGuess
        let [dx, dy] = [x - midX, y - midY]

        H00 = 2 * r ** 2 + D
        H01 = 2 * r * dx
        H02 = 2 * r * dy

        H11 = 2 * dx ** 2 - D
        H12 = 2 * dx * dy

        H22 = 2 * dy ** 2 - D

        let hessianComp = [
            [H00, H01, H02],
            [H01, H11, H12],
            [H02, H12, H22]
        ];

        return math.matrix(hessianComp)
    }

    hessian() {
        // Hessian of the objective function at current guess.
        let hessian = math.zeros(3, 3)

        for (let point of this.points) {
            let hessianPart = this.hessianContribution(point);
            hessian = math.add(hessian, hessianPart);
        }

        hessian = math.multiply(hessian, 4)

        return hessian
    }

    newton() {
        /*
        Ordinary Newton's method
		*/

        // Flag, that tells us if we have converged yet
        this.newtonConverged = false;
        this.iterationCount = 0

        for (let iteration of [...Array(MAX_ITERATIONS).keys()]) {
            // Main Iteration

            // Finding current Hessian and Gradient
            let H = this.hessian();
            let F = math.multiply(this.gradient(), -1);

            // Solve for the next step
            let z = math.lusolve(H, F);
            z = z.toArray().flat()

            // Check if the step is tiny.
            if (norm(z) < TOLERANCE) {
                // Indicator of convergence
                this.newtonConverged = true;
                this.iterationCount = iteration
                break;
            } else if (norm(z) > 10000) {
                // Indicator of divergence
                this.newtonConverged = false;
                this.iterationCount = iteration
                break
            }

            // Apply next step
            this.bestGuess = math.add(this.bestGuess, z);

            // check if radius is not too small
            if (this.bestGuess[0] < 0.1) {
            	// Indicator of divergence
                this.newtonConverged = false;
                this.iterationCount = iteration
                break
            }

            // repeat loop

        }
    }
}

class Finder extends Solver {
    /*
    Here we add all the checker methods to the Solver object.
    */

    devianceCostomCircle(point, circle) {
        /*
        Calculates the error contribution from
        the given point to any circle
        */
        let [r, midX, midY] = circle;
        let [x, y] = point;

        return r ** 2 - ((x - midX) ** 2 + (y - midY) ** 2)
    };

    objectiveFunction(circle) {
        // Gets the value of the objective function for any circle
        let value = 0;

        for (let point of this.points) {
            value += this.devianceCostomCircle(point, circle) ** 2
        }

        return value
    }

    initialObjective() {
        // value of objective function at initial circle
        return this.objectiveFunction(this.initialGuess);
    }

    currentObjective() {
        // value of objective function at current circle
        return this.objectiveFunction(this.bestGuess);
    }

    randomiseGuess() {
        /*
        This method randomises the best guess,
        so it can be used as a new seed for
        Newton's, should Newton's diverge.
        */
        let [r, midX, midY] = this.initialGuess

        r = math.random(0.1 * r + 0.1, 50 * r + 1);
        midX = math.random(0.1 * midX, 2 * midX);
        midY = math.random(0.1 * midY, 2 * midY);

        this.bestGuess = [r, midX, midY];
    }

    findBestCircle() {
        /*
        Combines all methods and tries to find the best circle
        */

        // Fist we check if the initial guess is good enough
        if (this.initialObjective() < TOLERANCE) {
            this.bestGuess = [...this.initialGuess];
            return 0
        }

        // Run Newton's until it converges or return initial guess
        let counter = 0
        while (counter <= MAX_NEWTON_ITERATIONS) {
            this.newton()

            // Check if newton converged
            if (this.newtonConverged == true || this.currentObjective() < TOLERANCE) {
                return 0
            } else {
                this.randomiseGuess()
                counter += 1
            }
        }

        this.bestGuess = [...this.initialGuess]

    }
}

class App {

    constructor(canvasContainerID, evaluateButtonID) {
        // finding canvas container node
        this.canvasContainer = document.getElementById(canvasContainerID);

        // finding evaluate button node
        this.evaluateButton = document.getElementById(evaluateButtonID);

        // attributes for handling paths
        this.stroking = false;
        this.paths = [];
        this.path = [];

        // attributes for handling circles
        this.foundCircle = false;
        this.circle = [50, 100, 100];

        // create canvas node
        this.canvas = document.createElement("canvas");

        // insert in DOM (inside container)
        this.canvasContainer.appendChild(this.canvas);

        // creating context
        this.ctx = this.canvas.getContext("2d");

        // resize canvas to proper size
        this.resizeCanvas();

    };

    resizeCanvas() {

        //  find current container dimensions
        this.containerDimensions = this.canvasContainer.getBoundingClientRect();

        // Save current canvas state
        let save = this.ctx.getImageData(0, 0,
            this.containerDimensions.width, this.containerDimensions.height)

        // set canvas dimensions
        this.canvas.width = this.containerDimensions.width;
        this.canvas.height = this.containerDimensions.height;

        // restore canvas state
        this.ctx.putImageData(save, 0, 0);


    };


    getCanvasCoord(event) {
        // get current dimensions
        this.containerDimensions = this.canvasContainer.getBoundingClientRect();

        // returns list of coord. relative to 
        // canvas origin
        let [canvasX, canvasY] = [
            this.containerDimensions.x,
            this.containerDimensions.y
        ];

        let [pageX, pageY] = [
            event.pageX,
            event.pageY
        ];

        return [pageX - canvasX, pageY - canvasY]
    }

    drawCircle() {
    	// first check if there is a circle to draw
    	if(this.foundCircle) {

    		this.ctx.strokeStyle = "#FF0000";
    		let [r, x, y] = this.circle
    		this.ctx.beginPath();
    		this.ctx.arc(x, y, r, 0 , 2*Math.PI);

    		this.ctx.stroke()

    	}
    }
    drawPath(path) {
        /*
		Draws a list of points on the canvas
    	*/

        // check if path is non empty
        if (path == []) {
            return false;
        };

        // being path and move to beginning
        this.ctx.strokeStyle = "#F0EAD6";

        let [x, y] = path[0];
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);

        // loop over all points
        for (let point of path) {

            // draw point
            let [x, y] = point;
            this.ctx.lineTo(x, y);
        }

        // stroke path
        this.ctx.stroke()
    };

    drawPaths() {
        /*
		draw all current paths
    	*/

        // Check if there is a currently stroked path
        if (this.path != []) {
            this.drawPath(this.path)
        }

        // check if there are any saved paths
        if (this.paths != []) {

            // Draw each path
            for (let path of this.paths) {
                this.drawPath(path)
            }
        }

    };

    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    };

    clearApp() {
    	// resets everything
    	this.foundCircle = false;
    	this.path = [];
    	this.paths = [];

    };

    draw() {
        /*
		function that draws on the canvas
    	*/

        // clear canvas
        this.clearCanvas()

        // drawing paths and circles
        this.drawPaths()
        this.drawCircle()

    };


    startPath(event) {
        // start new path
        this.stroking = true
    };

    continuePath(event) {
        // push coord. into current path
        // if we are stroking
        if (this.stroking) {
            let [x, y] = this.getCanvasCoord(event);
            this.path.push([x, y]);

            // draw
            this.draw()
        }


    };

    endPath(event) {
        // end path if we are stroking
        if (this.stroking) {

            this.stroking = false;
            this.paths.push(this.path);

            this.path = []
        }
    }

    findCricle() {
    	
    	// get list of points
    	let points = this.paths.flat()

    	// create Finder Object
    	let finder = new Finder(points)

    	// find best circle
    	finder.findBestCircle()

    	// save it as our circle
    	this.circle = finder.bestGuess;
    	this.foundCircle = true;
    	console.log(finder.newtonConverged)
    	console.log(this.circle)
    	this.drawCircle()
    }

    addListeners() {
        // binding this for all handlers
        this.startPath = this.startPath.bind(this);
        this.continuePath = this.continuePath.bind(this);
        this.endPath = this.endPath.bind(this);
        this.resizeCanvas = this.resizeCanvas.bind(this);
        this.draw = this.draw.bind(this);
        this.findCricle = this.findCricle.bind(this);

        // handle mouse down
        this.canvasContainer.addEventListener("mousedown",
            this.startPath);

        // handle mouse move
        this.canvasContainer.addEventListener("mousemove",
            this.continuePath);

        // handle mouse up
        this.canvasContainer.addEventListener("mouseup",
            this.endPath);

        // handle resize
        window.addEventListener("resize",
            this.resizeCanvas);
        window.addEventListener("orientationchange",
            this.resizeCanvas);

        // handle click on evaluate button
        this.evaluateButton.addEventListener("click", 
        	this.findCricle)

    }

    test() {
        // for debugging
        this.ctx.fillStyle = "red"
        this.ctx.fillRect(0, 0, 100, 100)
    };
};