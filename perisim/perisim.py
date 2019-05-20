import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import random

class PeriSim(nn.Module):
    '''Main class for the peristaltic table simulation'''

    def __init__(self, x, y, cargoPos, amplitude = 5., spacing = 80, stdDev = 40,
                 timeStep = 0.01, variance = 0.2, cargoVel = None, height = 1.,
                 cargoWeight = None, g = 9800, friction = 0.01, actForce = 100,
                 actTime = 0.1, gpu = False):
        self.defDict = {"x" : x, "y" : y, "amplitude" : amplitude, "spacing" : spacing, "stdDev" : stdDev, "timeStep" : timeStep,
         "variance" : variance, "height" : height, "g" : g, "friction" : friction, "actForce" : actForce, "actTime" : actTime}
        if gpu:
             self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        self.size = (x,y)
        self.variance = variance
        self.spacing = self.perturb(spacing)
        #print(self.size)
        tG = Variable(torch.ones(x,y), requires_grad=False).type(self.dtype)
        self.actuationPos = torch.nonzero(tG).type(self.dtype)*self.spacing
        self.amplitude = self.perturb(amplitude)
        if height == None:
            height = amplitude
        self.height = self.perturb(height)
        self.cellHeights = Variable(torch.zeros(x*y,1), requires_grad=False).type(self.dtype)+self.height
        self.ts = timeStep
        self.actForce = actForce
        #self.a = (1/stdDev*math.sqrt(2*math.pi))
        self.b = self.perturb(2*(stdDev**2))
        self.cargoPos = Variable(torch.Tensor(cargoPos), requires_grad=True).type(self.dtype)
        self.grads = Variable(torch.zeros(self.cargoPos.size()), requires_grad=False).type(self.dtype)
        self.g = self.perturb(g)
        self.friction = self.perturb(friction)
        #Repeat Variables for multiplication
        self.actPosRepeat = self.actuationPos.repeat(self.cargoPos.size()[0],1).type(self.dtype)
        self.cellHeightRepeat = self.cellHeights.repeat(self.cargoPos.size()[0],1).type(self.dtype)
        self.time = 0.
        self.actTime = self.perturb(actTime/timeStep)
        self.forceApplied = Variable(torch.zeros(self.cargoPos.size()), requires_grad=False).type(self.dtype)
        if cargoVel == None:
            self.cargoVel = Variable(torch.zeros(self.cargoPos.size()), requires_grad=False).type(self.dtype)
        else:
            self.cargoVel = Variable(torch.Tensor(cargoVel), requires_grad=False).type(self.dtype)

        if cargoWeight == None:
            self.cargoWeight = Variable(torch.zeros(self.cargoPos.size()[0]), requires_grad=False).type(self.dtype)+10
        else:
            self.cargoWeight = Variable(torch.Tensor(cargoWeight), requires_grad=False).type(self.dtype)

    def perturb(self, x):
        '''Perturbs a variable within the noise level'''
        perturb = (random.random()-0.5)*2*x
        x = x + self.variance*perturb
        return x

    def actuate(self, x, y, direction):
        '''Changes the actuation level of a peristaltic cell'''
        if int(x)>=self.size[0] or int(y)>=self.size[1]:
            print("Linbot (%d, %d) not in array" % (x, y))
            return
        self.cellHeights[(x*self.size[0] + y)] = self.height + direction*self.amplitude
        self.actPosRepeat = self.actuationPos.repeat(self.cargoPos.size()[0],1)
        self.cellHeightRepeat = self.cellHeights.repeat(self.cargoPos.size()[0],1)
        actin = Variable(torch.Tensor([x,y]).repeat(self.cargoPos.size()[0],1))
        pinged = torch.nonzero(torch.sum(torch.abs(self.nearest_cell() - actin), 1) == 0)
        if len(pinged.size()) > 0:
            self.forceApplied.data[pinged.data.view(-1)] = self.actForce
        #print("Actuate (%d,%d) by %d" % (x,y,direction))

    def nearest_cell(self):
        '''Finds the closes peristaltic cell'''
        return torch.round(self.cargoPos/self.spacing)

    def update(self):
        '''Updates the surface  of the table and the cargo'''
        self.time = self.time+self.ts
        self.update_grad()

        angle = self.grads.atan()

        #v = v + at     a = F/m   F=Rsin(angle)-drag   R = -g*m*cos(angle)   v = v+gtcos(angle)sin(angle)
        drag = self.friction*self.cargoVel
        self.cargoVel = self.cargoVel+(-(self.g+self.forceApplied)*self.ts*angle.cos()*angle.sin())-drag
        self.cargoPos = self.cargoPos + self.cargoVel*self.ts
        self.forceApplied.data = self.forceApplied.data - self.actForce/self.actTime
        self.forceApplied.data[self.forceApplied.data<0] = 0

    def scale_tensor(self, tensor, scale):
        '''Increases size of tensor by scale by copying entries to the next
        entry (rather than the end) e.g scaleTensor([1,2,3],2) > [1,1,2,2,3,3]'''
        newTensor = tensor.view((tensor.size()[0],1,tensor.size()[1])).repeat(1, scale, 1).view((-1, tensor.size()[1]))
        return newTensor

    def partial_sum(self, tensor, length):
        '''Reduces a vecotr to length by summing over size/length
        e.g partialSum([1,2,3,4],2) > [3,7]'''
        newTensor = torch.sum(tensor.view(length, -1), dim=1).view(-1,1)
        return newTensor

    def update_grad(self):
        '''Update gradients based on actuations'''
        cargoPosRepeat = self.scale_tensor(self.cargoPos, self.actuationPos.size()[0]).type(self.dtype)
        dist = cargoPosRepeat-self.actPosRepeat.type(self.dtype)
        cargoHeights = self.cellHeightRepeat*torch.prod(torch.exp(-(dist * dist)/self.b), dim = 1).view(-1,1).type(self.dtype)
        gradsSplit = (-2*dist/self.b)*cargoHeights.view(cargoHeights.size()[0],1).expand(cargoHeights.size()[0],2).type(self.dtype)
        gradsX = self.partial_sum(gradsSplit[:, 0].contiguous(), self.cargoVel.size()[0]).type(self.dtype)
        gradsY = self.partial_sum(gradsSplit[:, 1].contiguous(), self.cargoVel.size()[0]).type(self.dtype)
        self.grads = torch.cat((gradsX,gradsY), dim=1).type(self.dtype)
        perturb = self.grads.data*((torch.rand(self.grads.size())*2)-1)
        self.grads.data = self.grads.data+(self.variance*perturb)
        # self.heights = torch.sum(torch.prod(gaussians,1).view(-1, actuationPos.size()[0]), dim=1)
        # print("gaussians")
        # print(gaussians)
        # print("heights")
        # print(self.heights)
        #
        # for i in range(cargoPos.size()[0]):
        #    self.heights[i].backward(retain_graph=True)
        #    self.grads.data[i] = cargoPos.grad.data[i]
    def update_weight(self):
        '''Updates the weights on the four cells surrounding each object'''
        gridspace = self.cargoPos/self.spacing
        #Make array of all laden cells
        maxXY = torch.ceil(gridspace)
        minXY = torch.floor(gridspace)
        maxXminY = torch.cat((maxXY[:,0].contiguous().view(-1,1),minXY[:,1].contiguous().view(-1,1)), dim = 1)
        minXmaxY = torch.cat((minXY[:,0].contiguous().view(-1,1),maxXY[:,1].contiguous().view(-1,1)), dim = 1)
        weightPos = torch.cat((maxXY, maxXminY, minXY, minXmaxY), dim = 0)
        #Find distance from the opposite side of the table from each cell to the ball (in grid space)
        weightDiffs = torch.abs(weightPos-gridspace.repeat(4,1))
        weightDiffs = 1-weightDiffs
        #Multiply both distances to give a proportion of weight on one cell
        weightDiffs = torch.prod(weightDiffs, dim=1)#
        #Multiply by the weight of each cargo to give the weight on each cell
        weightDistribution = weightDiffs*self.cargoWeight.repeat(4)
        weights = torch.cat((weightPos*self.spacing, weightDistribution.view(-1, 1)), dim = 1)
        return weights


    def heights(self, start, end, spacing = 1):
        '''Calculate height grid based on actuations'''
        gridSize = (math.ceil(end[0]-start[0]/spacing), math.ceil(end[1]-start[1]/spacing))
        tG = Variable(torch.ones(gridSize[0],gridSize[1]), requires_grad=False)
        offset = torch.Tensor([start[0], start[1]])
        testGrid = torch.nonzero(tG).float()
        testGrid.data = (testGrid.data*spacing) + offset
        # for i in range(gridSize[1]):
        #     for j in range(gridSize[0]):
        #         testGrid[(i*gridSize[0])+j] = torch.Tensor([(j*spacing)+start[0], (i*spacing)+start[1]])
        testGridRepeat = self.scale_tensor(testGrid, self.actuationPos.size()[0])
        actPosGridRepeat = self.actuationPos.repeat(testGrid.size()[0],1)
        cellHeightGridRepeat = self.cellHeights.repeat(testGrid.size()[0],1)

        dist = testGridRepeat-actPosGridRepeat.float()
        splitHeights = cellHeightGridRepeat*torch.prod(torch.exp(-(dist * dist)/self.b), dim = 1).view(-1,1)
        gridHeights = self.partial_sum(splitHeights, testGrid.size()[0])
        grid = torch.cat((testGrid, gridHeights), dim =1)
        return grid, gridSize

    def visualise(self, xmin = None, ymin = None, xmax = None, ymax = None, cmap=None):
        from mayavi import mlab
        if xmin is None:
            xmin = -self.spacing
        if ymin is None:
            ymin = -self.spacing
        if xmax is None:
            xmax = (self.size[0])*self.spacing
        if ymax is None:
            ymax = (self.size[1])*self.spacing

        grid, gSize = self.heights((xmin,ymin), (xmax,ymax))

        surf = mlab.mesh(grid.data[:,0].contiguous().view(gSize).numpy(),
                               grid.data[:,1].contiguous().view(gSize).numpy(),
                               grid.data[:,2].contiguous().view(gSize).numpy(), color=(1,0.5,0))

        objH, _ = self.heights((self.cargoPos.data[0,0], self.cargoPos.data[0,1]), (self.cargoPos.data[0,0]+1, self.cargoPos.data[0,1]+1))

        pts = mlab.points3d(self.cargoPos.data[0,0], self.cargoPos.data[0,1], objH.data[0,2]+25, 50, scale_factor=1, color=(1,1,1))
        mlab.show()
