import turtle as t
from turtletools import turtleTools
import math as m
import random as r
import time
import argparse
t.setup(960, 720)
t.colormode(255)
t.title("nerwork1")
turtools = turtleTools(t.getcanvas(), -240, -180, 240, 180, True)
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="supply file")
parser.add_argument("-tps", "--tps", help="fix tps")
parser.add_argument("-fps", "--fps", help="print fps", action="store_true")
args = parser.parse_args()
t.hideturtle()
# t.screensize(400, 3000)
class main:
    def __init__(self):
        self.sp = t.Turtle(shape = "circle")
        self.sp.hideturtle()
        self.sp.pencolor(0, 0, 0)
        self.sp.penup()

        self.nodes = [] # list of nodes (2D array)
        self.weights = [] # list of weights (3D array)
        self.weightedSums = [] # list of weighted sums (2D array, just node values reversed through activation function with the first layer omitted)
        self.gradient = [] # gradient matrix (2D array)
        self.stochasticGrad = [] # gradient matrix of backprop results from multiple samples (3D array)
        self.bias = [] # list of biases (2D array)
        self.format = [] # list for formatting drawing

        self.debug = False # testing

        self.sample = 0
        self.data = [] # training data [2D array]
        self.pres = [] # list loaded in with a sample determining the expected result of that sample
        if args.filename == None: # training data filename
            self.filename = "Not Found"
            self.file = None
            
        else:
            self.filename = args.filename
            try:
                self.file = open(self.filename, "r") # training data file (csv)
            except:
                print("Error: file " + args.filename + " not found")
                return
            try:
                raw = self.file.read().split(',')
            except:
                print("Error: file " + args.filename + " could not be read")
                return
            iter = 784
            while (iter + 1) < len(raw):
                sublist = []
                sublist.append(raw[iter][-1])
                iter += 1
                for i in range(28 * 28 - 1):
                    if iter >= len(raw):
                        break
                    sublist.append(int(raw[iter]) / 255) # divide by 255 to get a value from 0 to 1
                    iter += 1
                if iter >= len(raw): 
                    break
                i = 0
                acc = "" # last number of grid and first label of next data point are separated by a newline not a comma
                while raw[iter][i] != '\n' and i < len(raw[iter]):
                    acc += raw[iter][i]
                    i += 1
                sublist.append(int(acc) / 255)
                self.data.append(sublist)
            print("file " + self.filename + " successfully loaded, " + str(len(self.data)) + " samples parsed")
        t.setworldcoordinates(-240, -180, 240, 180)
    def loadData(self, instance, render=True, printf=True): # load a sample (instace) into the network
        try:
            instance = int(instance)
            dummy = self.data[instance]
        except:
            print("No sample " + str(instance))
            return
        if len(self.nodes[0]) < (len(self.data[instance]) - 1):
            print("Error: not enough layer 1 nodes")
            return
        for i in range(1, len(self.data[instance])):
            self.nodes[0][((i - 1) % 28) * 28 + int((i - 1) / 28)] = self.data[instance][i]
            '''
            because my renderer goes like
            1  5  9  13
            2  6  10 14
            3  7  11 15
            4  8  12 16

            and the data is like
            1  2  3  4
            5  6  7  8
            9 10 11 12
            13 14 15 16

            some translation calculations must take place
            specifically, switching mod and divison
            '''
        self.process()
        if render:
            self.drawNetwork(True, False)
        self.sample = instance
        self.pres = []
        for i in range(len(self.nodes[-1])): # setup the pres list
            self.pres.append(0)
        self.pres[int(self.data[self.sample][0])] = 1
        if printf:
            print("Sample " + str(instance) + " loaded (" + self.data[instance][0] + ")")
    def setup(self, layers, nodesPerLayer, initWeights=[], initBias=[]): # setup the nodes, weights, biases, etc for the network (randomised values for weights and biases)
        self.nodes = [] # list of nodes (2D array)
        self.weights = [] # list of weights (3D array)
        self.weightedSums = [] # list of weighted sums (2D array, just node values reversed through activation function with the first layer omitted)
        self.gradient = [] # gradient matrix (2D array)
        self.stochasticGrad = [] # gradient matrix of backprop results from multiple samples (3D array)
        self.bias = [] # list of biases (2D array)
        # self.format = [] # list for formatting drawing
        if layers != len(nodesPerLayer):
            print("setup layer mismatch!")
            return
        for i in range(layers):
            self.nodes.append([])
            self.weightedSums.append([])
            self.gradient.append([])
            self.weights.append([])
            self.bias.append([])
            for j in range(nodesPerLayer[i]):
                self.weights[i].append([])
                self.nodes[i].append(0)
                self.weightedSums[i].append(0)
                self.gradient[i].append(0) # add bias
                if i > 0:
                    self.bias[i].append((r.random() - 0.5) * 1)
                if i < (layers - 1):
                    for k in range(nodesPerLayer[i + 1]):
                        self.gradient[i].append(0) # add weights
                        self.weights[i][j].append((r.random() - 0.5) * 1)
        self.gradient.pop(-1)
        self.weights.pop(-1)
    def setinp(self): # set first layer of nodes to 0
        for i in range(len(self.nodes[0])):
            self.nodes[0][i] = 0
        return
    def drawNetwork(self, nodeValues=True, wires=True):
        self.sp.clear()
        initPositions = []
        size = 22
        tsize = size / 20
        if len(self.format) < 1: # regenerate format list if it is empty with default settings
            self.format.append('Left')
        totalXlen = 0
        for i in range(len(self.nodes)):
            try:
                dummy = self.format[i + 1] 
            except:
                self.format.append(1)
                self.format.append(len(self.nodes[i]))
            totalXlen += int(len(self.nodes[i]) / self.format[i + 1]) * size * 1.1
            totalXlen += size * 2
        x = -totalXlen / 2 * 0.5
        maxY = 0
        for i in range(len(self.nodes)):
            initPositions.append(x)
            initPositions.append(size * 0.275 * (self.format[i + 1] - 1))
            x += (size * 0.55 * (int(len(self.nodes[i]) / self.format[i + 1]) - 1)) + size * 2
            if initPositions[-1] > maxY:
                maxY = initPositions[-1]
        t.screensize(totalXlen + 100, maxY * 4 + 100)
        if wires:
            self.sp.pensize(1)
            for i in range(len(self.nodes) - 1):
                for j in range(len(self.nodes[i])):
                    for k in range(len(self.weights[i][j])):
                        col = int(255 - abs(255 - round(255 / (1 + m.e ** (-self.weights[i][j][k])) % 255) - 127.5) * 2) # uses shifted sigmoid for weight 'weights' (how dark the connection appears when drawn as a function of how large it is)
                        self.sp.pencolor(col, col, col)
                        self.sp.goto(initPositions[i * 2] + size * 0.55 * int(j / self.format[i + 1]), initPositions[i * 2 + 1] - size * 0.55 * (j % self.format[i + 1]))
                        self.sp.pendown()
                        self.sp.goto(initPositions[i * 2 + 2] + size * 0.55 * int(k / self.format[i + 2]), initPositions[i * 2 + 3] - size * 0.55 * (k % self.format[i + 2]))
                        self.sp.penup()
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
            # for j in range(int(len(self.nodes[i]) / 10)):
                x = initPositions[i * 2] + size * 0.55 * int(j / self.format[i + 1])
                y = initPositions[i * 2 + 1] - size * 0.55 * (j % self.format[i + 1])
                self.sp.goto(x, y)
                self.sp.turtlesize(tsize)
                self.sp.color(0, 0, 0)
                self.sp.stamp()
                self.sp.turtlesize(tsize * 0.8)
                col = round(255 * self.nodes[i][j])
                self.sp.color(col, col, col)
                self.sp.stamp()
                col = (col + 127) % 255
                self.sp.pencolor(col, col, col)
                self.sp.goto(x + size * 0.025, y - size * 0.25)
                if nodeValues:
                    try:
                        if self.nodes[i][j] < 0.1:
                            char1 = '0'
                        else:
                            char1 = str(self.nodes[i][j])[2]
                    except:
                        char1 = '0'
                    try:
                        if self.nodes[i][j] < 0.01:
                            char2 = '0'
                        else:
                            char2 = str(self.nodes[i][j])[3]
                    except:
                        char2 = '0'
                    self.sp.write(char1 + char2, move=False, align='center', font=('Courier', int(size * 0.5), 'bold'))
        t.Screen().update()
    def activationFunction(self, inp, printf=True): #sometimes called "transfer function"
        try:
            return 1 / (1 + m.e ** (-inp)) # sigmoid function
        except:
            if printf:
                print("Activation: Result too large (default to 0)")
            return 0
    def derivAF(self, inp, printf=True): # derivative of the activation function
        try:
            sig = 1 / (1 + m.e ** (-inp))
            return sig * (1 - sig) # derivative of sigmoid function
        except:
            if printf:
                print("derivAF: Result too large (default to 0)")
            return 0
    def calculateCost(self, printf=False): # calculate cost relative to the presumed correct response (self.pres)
        if len(self.nodes[-1]) != len(self.pres):
            return None
        acc = 0
        for i in range(len(self.nodes[-1])):
            acc += (self.nodes[-1][i] - self.pres[i]) * (self.nodes[-1][i] - self.pres[i])
        if printf:
            print("Cost: " + str(acc))
        return acc
    def backProp(self): # backpropagate, outputs a list self.gradient
        self.lastLayer = [] # 1D list containing all of the derivatives of a particular layer (the last refers to the last computation which moves from the last layer in the network to the first as we backpropagate)
        for i in range(len(self.nodes[-1])): # load derivatives of the output layer
            self.lastLayer.append(2 * (self.nodes[-1][i] - self.pres[i]))
        layers = len(self.nodes)
        for i in range(layers - 1, 0, -1): # do layers - 1 cycles, starting at layers - 1 and ending at 1
            for j in range(len(self.nodes[i])):
                derivAcivate = self.derivAF(self.weightedSums[i][j])
                llj = self.lastLayer[j]
                for k in range(len(self.nodes[i - 1])):
                    self.gradient[i - 1][j * (len(self.nodes[i - 1]) + 1) + k] = self.nodes[i - 1][k] * derivAcivate * llj # set weight
                self.gradient[i - 1][(j + 1) * (len(self.nodes[i - 1]) +  1) - 1] = 1 * derivAcivate * llj #set bias

                # gradient and weight lists have lengths of layers - 1, reflecting how there are more layers of nodes than layers of weights. The bias list index matches that of the nodes, but index 0 of the bias list is empty
                # self.weight[i - 1][k][j] represents self.gradient[i - 1][j * (len(self.nodes[i - 1]) + 1) + k], which is a connection from self.nodes[i - 1][k] to self.nodes[i][j]

            if i != 1:
                lastLayer2 = self.lastLayer.copy()
                self.lastLayer = [] # setup lastLayer for next backprop iteration (only happens layers - 2 times)
                for j in range(len(self.nodes[i - 1])):
                    acc = 0
                    for k in range(len(self.nodes[i])):
                        acc += self.weights[i - 1][j][k] * self.derivAF(self.weightedSums[i][k]) * lastLayer2[k]
                    self.lastLayer.append(acc)
        
        return self.gradient
    def adjustWeightsAndBias(self, adjuster, scale=1): # adjust weights and biases
        for h in range(len(adjuster)):
            for i in range(1, len(self.nodes)):
                for j in range(len(self.nodes[i])):
                    for k in range(len(self.nodes[i - 1])):
                        self.weights[i - 1][k][j] -= scale * (len(adjuster) / len(self.data)) * adjuster[h][i - 1][j * (len(self.nodes[i - 1]) + 1) + k]
                    self.bias[i][j] -= scale * (len(adjuster) / len(self.data)) * adjuster[h][i - 1][(j + 1) * (len(self.nodes[i - 1]) + 1) - 1]
    def calculateTotalCost(self, printf=True):
        acc = 0
        for i in range(len(self.data)):
            self.loadData(i, False)
            acc += self.calculateCost()
        if printf:
            print("average cost: " + str(acc / len(self.data)))
        return acc
    def stochasticBackProp(self, samples): # takes in a list of training samples (labeled 0 - 9999 in MNIST set)
        self.stochasticGrad = []
        for i in range(len(samples)):
            self.loadData(samples[i], False)
            self.stochasticGrad.append(self.backProp())
        self.adjustWeightsAndBias(self.stochasticGrad)
    def stochasticTrain(self, samplesPerIter, iterations, show=False): # fully train the model, choose the number of samples per iteration and the number of iterations. After this script runs the model will be fully trained
        for i in range(iterations):
            available = []
            for j in range(len(self.data)):
                available.append(j)
            sample = []
            for k in range(samplesPerIter):
                choice = r.randint(0, len(available) - 1)
                sample.append(available[choice])
                available.pop(choice)
            self.stochasticBackProp(sample)
            self.calculateCost(True)
            if show:
                self.process()
                self.drawNetwork(False, False)
            if i % 100 == 0:
                self.save()
    def process(self): # calculate neural network node values
        for i in range(1, len(self.nodes)):
            for j in range(len(self.nodes[i])):
                acc = 0
                for k in range(len(self.nodes[i - 1])):
                    acc += self.weights[i - 1][k][j] * self.nodes[i - 1][k] # calculate weighted sum
                acc += self.bias[i][j] # add bias
                self.weightedSums[i][j] = acc
                self.nodes[i][j] = self.activationFunction(acc) # run weighted sum + bias through activation function
    def save(self): # save weights and biases to file
        current = round(time.time())
        f = open(str(current) + ".txt", "x")
        write1 = str(len(self.nodes))
        for i in range(len(self.nodes)):
            write1 += ' ' + str(len(self.nodes[i]))
        f.write(write1 + ' ' + '\n')
        f.write("Weights: \n")
        for i in range(1, len(self.nodes)):
            for j in range(len(self.nodes[i])):
                for k in range(len(self.nodes[i - 1])):
                    f.write(str(self.weights[i - 1][k][j]) + ' ')
                f.write('\n')
            f.write('\n')
        f.write('Biases: \n')
        for i in range(1, len(self.nodes)):
            for j in range(len(self.nodes[i])):
                f.write(str(self.bias[i][j]) + ' ')
            f.write('\n')
        f.close()
        print("successfully saved to " + str(current) + ".txt")
        return
    def load(self, filename): # load weights and biases from file
        try:
            f = open(filename, "r")
        except:
            print("Error: file " + filename + " not found")
            return
        try:
            read1 = f.read().split(' ')
        except:
            print("Error: file " + filename + " could not be read")
            return
        weightPtr = 0
        nodesPerLayer = []
        for i in range((int(read1[weightPtr]))):
            weightPtr += 1
            nodesPerLayer.append(int(read1[weightPtr]))
        weightPtr += 2
        print("Loaded " + str(len(nodesPerLayer)) + " layers with " + str(nodesPerLayer) + " nodes per layer")
        self.setup(len(nodesPerLayer), nodesPerLayer)
        biasPtr = weightPtr
        while biasPtr < len(read1) - 1 and read1[biasPtr] != "\n\nBiases:":
            biasPtr += 1
        biasPtr += 1
        adjuster = []
        for i in range(1, len(self.nodes)):
            adjuster.append([])
            for j in range(len(self.nodes[i])):
                for k in range(len(self.nodes[i - 1])):
                    adjuster[i - 1].append(float(read1[weightPtr]))
                    weightPtr += 1
                adjuster[i - 1].append(float(read1[biasPtr]))
                biasPtr += 1
        for i in range(1, len(self.nodes)):
                for j in range(len(self.nodes[i])):
                    for k in range(len(self.nodes[i - 1])):
                        self.weights[i - 1][k][j] = adjuster[i - 1][j * (len(self.nodes[i - 1]) + 1) + k]
                        self.bias[i][j] = adjuster[i - 1][(j + 1) * (len(self.nodes[i - 1]) + 1) - 1]
        print("Loaded weights and biases from " + filename)

obj = main()
layers = 4
nodesPerLayer = [28 * 28, 16, 16, 10]
obj.format = ['Left', 28, 16, 16, 10]
obj.setup(layers, nodesPerLayer)
def draw(values=True, wires=False, proc=True): # draw the neural network
    if proc:
        obj.process()
    obj.drawNetwork(values, wires)
def showRandom(): # show a random sample
    sample = r.randint(0, len(obj.data) - 1)
    obj.loadData(sample)
# draw()
obj.load("fullTake1.txt")
showRandom()
# while True: # script to manually look at samples
#     sample = input("load sample: ")
#     if sample == "quit" or sample == "q":
#         break
#     obj.loadData(sample)

# while True: # script to slideshow through random samples
#     t.ontimer(showRandom(), 500)

# obj.calculateTotalCost()
# obj.save()

# count = 0
# while True:
#     obj.loadData(r.randint(0, len(obj.data) - 1), False, False)
#     obj.backProp()
#     obj.adjustWeightsAndBias([obj.gradient], 5000)
#     obj.calculateCost(True)
#     if count % 100 == 0:
#         draw()
#     if count % 10000 == 0:
#         obj.save()
#     if turtools.keyPressed('space'):
#         break
#     count += 1

# for i in range(5):
#     obj.process()
#     obj.backProp()
#     obj.adjustWeightsAndBias([obj.gradient], 10000)
#     obj.calculateCost(True)
#     # draw()
# draw()

# obj.save()
# obj.stochasticTrain(100, 10, True)
# obj.save()

keys = [] #script to slideshow when keys is pressed
print("press space to display a random sample")
mouseSize = 10
while True:
    if turtools.keyPressed('space'):
        if keys.count('space') == 0:
            keys.append('space')
            showRandom()
            obj.calculateCost(True)
    else:
        if keys.count('space') > 0:
            keys.remove('space')
    if turtools.keyPressed('s'): # save the values if s is pressed
        if keys.count('s') == 0:
            keys.append('s')
            obj.save()
    else:
        if keys.count('s') > 0:
            keys.remove('s')
    if turtools.keyPressed('c'): # clear the screen if c is pressed
        if keys.count('c') == 0:
            keys.append('c')
            obj.setinp()
            draw()
    else:
        if keys.count('c') > 0:
            keys.remove('c')
    if turtools.mouseDown(): # draw if mouse is pressed
        if keys.count('mouse') == 0:
            keys.append('mouse')
    else:
        if keys.count('mouse') > 0:
            keys.remove('mouse')
    if keys.count('mouse') > 0: # draw your own number script
        mx, my = turtools.getMouseCoords()
        # print(mx, my) # each node is 12 by 12, origin (0, 0 node) at -235.404, 167.657
        # but bottom left node is at -235.915, -159.429
        if mx > -240 and mx < 97 and my > -165 and my < 173:
            obj.nodes[0][round((mx + 235.404) / 12) * 28 + (27 - round((my + 159.429) / 12))] = 0.99
        draw(False, False, True)
    t.forward(0)
    t.Screen().update()

t.mainloop()
t.bye()