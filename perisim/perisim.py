import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import random

class PeriSim(nn.Module):
    '''Main class for the peristaltic table simulation'''

    def __init__(self, x, y, cargo_pos, amplitude=5., spacing=80, stddev=40,
                 time_step=0.01, variance=0.2, cargo_vel=None, height=1.,
                 cargo_weight=None, g=9800, friction=0.01, act_force=100,
                 act_time=0.1, gpu=False):
        self.defDict = {"x" : x, "y" : y, "amplitude" : amplitude, "spacing" : spacing, "stddev" : stddev, "time_step" : time_step,
         "variance" : variance, "height" : height, "g" : g, "friction" : friction, "act_force" : act_force, "act_time" : act_time}
        if gpu:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        self.size = (x, y)
        self.variance = variance
        self.spacing = self.perturb(spacing)
        tG = Variable(torch.ones(x, y), requires_grad=False).type(self.dtype)
        self.actuation_pos = torch.nonzero(tG).type(self.dtype) * self.spacing
        self.amplitude = self.perturb(amplitude)
        if height is None:
            height = amplitude
        self.height = self.perturb(height)
        self.cell_heights = Variable(torch.zeros(x * y, 1), requires_grad=False).type(self.dtype) + self.height
        self.ts = time_step
        self.act_force = act_force
        self.b = self.perturb(2 * (stddev ** 2))
        self.cargo_pos = Variable(torch.Tensor(cargo_pos), requires_grad=True).type(self.dtype)
        self.grads = Variable(torch.zeros(self.cargo_pos.size()), requires_grad=False).type(self.dtype)
        self.g = self.perturb(g)
        self.friction = self.perturb(friction)

        #Repeat Variables for multiplication
        self.act_pos_repeat = self.actuation_pos.repeat(self.cargo_pos.size()[0], 1).type(self.dtype)
        self.cell_height_repeat = self.cell_heights.repeat(self.cargo_pos.size()[0], 1).type(self.dtype)
        self.time = 0.
        self.act_time = self.perturb(act_time / time_step)
        self.force_applied = Variable(torch.zeros(self.cargo_pos.size()), requires_grad=False).type(self.dtype)
        if cargo_vel is None:
            self.cargo_vel = Variable(torch.zeros(self.cargo_pos.size()), requires_grad=False).type(self.dtype)
        else:
            self.cargo_vel = Variable(torch.Tensor(cargo_vel), requires_grad=False).type(self.dtype)

        if cargo_weight is None:
            self.cargo_weight = Variable(torch.zeros(self.cargo_pos.size()[0]), requires_grad=False).type(self.dtype) + 10
        else:
            self.cargo_weight = Variable(torch.Tensor(cargo_weight), requires_grad=False).type(self.dtype)

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
        self.cell_heights[(x * self.size[0] + y)] = self.height + direction * self.amplitude
        self.act_pos_repeat = self.actuation_pos.repeat(self.cargo_pos.size()[0], 1)
        self.cell_height_repeat = self.cell_heights.repeat(self.cargo_pos.size()[0], 1)
        actin = Variable(torch.Tensor([x,y]).repeat(self.cargo_pos.size()[0], 1))
        pinged = torch.nonzero(torch.sum(torch.abs(self.nearest_cell() - actin), 1) == 0)
        if len(pinged.size()) > 0:
            self.force_applied.data[pinged.data.view(-1)] = self.act_force

    def nearest_cell(self):
        '''Finds the closes peristaltic cell'''
        return torch.round(self.cargo_pos / self.spacing)

    def update(self):
        '''Updates the surface  of the table and the cargo'''
        self.time = self.time+self.ts
        self.update_grad()

        angle = self.grads.atan()

        drag = self.friction*self.cargo_vel
        self.cargo_vel = self.cargo_vel + (-(self.g + self.force_applied) * self.ts * angle.cos() * angle.sin()) - drag
        self.cargo_pos = self.cargo_pos + self.cargo_vel * self.ts
        self.force_applied.data = self.force_applied.data - self.act_force / self.act_time
        self.force_applied.data[self.force_applied.data < 0] = 0

    def scale_tensor(self, tensor, scale):
        '''Increases size of tensor by scale by copying entries to the next
        entry (rather than the end) e.g scaleTensor([1,2,3],2) > [1,1,2,2,3,3]'''
        new_tensor = tensor.view((tensor.size()[0], 1, tensor.size()[1])).repeat(1, scale, 1).view((-1, tensor.size()[1]))
        return new_tensor

    def partial_sum(self, tensor, length):
        '''Reduces a vecotr to length by summing over size/length
        e.g partialSum([1,2,3,4],2) > [3,7]'''
        new_tensor = torch.sum(tensor.view(length, -1), dim=1).view(-1, 1)
        return new_tensor

    def update_grad(self):
        '''Update gradients based on actuations'''
        cargo_pos_repeat = self.scale_tensor(self.cargo_pos, self.actuation_pos.size()[0]).type(self.dtype)
        dist = cargo_pos_repeat-self.act_pos_repeat.type(self.dtype)
        cargo_heights = self.cell_height_repeat * torch.prod(torch.exp(-(dist * dist) / self.b), dim=1).view(-1, 1).type(self.dtype)
        grads_split = (-2*dist/self.b)*cargo_heights.view(cargo_heights.size()[0], 1).expand(cargo_heights.size()[0], 2).type(self.dtype)
        grads_x = self.partial_sum(grads_split[:, 0].contiguous(), self.cargo_vel.size()[0]).type(self.dtype)
        grads_y = self.partial_sum(grads_split[:, 1].contiguous(), self.cargo_vel.size()[0]).type(self.dtype)
        self.grads = torch.cat((grads_x,grads_y), dim=1).type(self.dtype)
        perturb = self.grads.data*((torch.rand(self.grads.size())*2)-1)
        self.grads.data = self.grads.data+(self.variance*perturb)

    def update_weight(self):
        '''Updates the weights on the four cells surrounding each object'''
        gridspace = self.cargo_pos / self.spacing
        #Make array of all laden cells
        maxXY = torch.ceil(gridspace)
        minXY = torch.floor(gridspace)
        maxXminY = torch.cat((maxXY[:, 0].contiguous().view(-1, 1), minXY[:, 1].contiguous().view(-1, 1)), dim=1)
        minXmaxY = torch.cat((minXY[:, 0].contiguous().view(-1, 1), maxXY[:, 1].contiguous().view(-1, 1)), dim=1)
        weight_pos = torch.cat((maxXY, maxXminY, minXY, minXmaxY), dim=0)
        #Find distance from the opposite side of the table from each cell to the ball (in grid space)
        weight_diffs = torch.abs(weight_pos-gridspace.repeat(4, 1))
        weight_diffs = 1-weight_diffs
        #Multiply both distances to give a proportion of weight on one cell
        weight_diffs = torch.prod(weight_diffs, dim=1)
        #Multiply by the weight of each cargo to give the weight on each cell
        weight_distribution = weight_diffs*self.cargo_weight.repeat(4)
        weights = torch.cat((weight_pos*self.spacing, weight_distribution.view(-1, 1)), dim=1)
        return weights


    def heights(self, start, end, spacing = 1):
        '''Calculate height grid based on actuations'''
        grid_size = (math.ceil(end[0]-start[0]/spacing), math.ceil(end[1]-start[1]/spacing))
        tG = Variable(torch.ones(grid_size[0], grid_size[1]), requires_grad=False)
        offset = torch.Tensor([start[0], start[1]])
        test_grid = torch.nonzero(tG).float()
        test_grid.data = (test_grid.data*spacing) + offset
        test_grid_repeat = self.scale_tensor(test_grid, self.actuation_pos.size()[0])
        act_pos_grid_repeat = self.actuation_pos.repeat(test_grid.size()[0], 1)
        cell_height_grid_repeat = self.cell_heights.repeat(test_grid.size()[0], 1)

        dist = test_grid_repeat-act_pos_grid_repeat.float()
        split_heights = cell_height_grid_repeat*torch.prod(torch.exp(-(dist * dist)/self.b), dim=1).view(-1, 1)
        grid_heights = self.partial_sum(split_heights, test_grid.size()[0])
        grid = torch.cat((test_grid, grid_heights), dim=1)
        return grid, grid_size

    def visualise(self, xmin=None, ymin=None, xmax=None, ymax=None):
        from mayavi import mlab

        if xmin is None:
            xmin = -self.spacing
        if ymin is None:
            ymin = -self.spacing
        if xmax is None:
            xmax = (self.size[0])*self.spacing
        if ymax is None:
            ymax = (self.size[1])*self.spacing

        grid, g_size = self.heights((xmin, ymin), (xmax, ymax))

        mlab.mesh(grid.data[:, 0].contiguous().view(g_size).numpy(),
                               grid.data[:, 1].contiguous().view(g_size).numpy(),
                               grid.data[:, 2].contiguous().view(g_size).numpy(), color=(1, 0.5, 0))

        obj_h, _ = self.heights((self.cargo_pos.data[0, 0], self.cargo_pos.data[0, 1]), (self.cargo_pos.data[0, 0] + 1, self.cargo_pos.data[0, 1] + 1))

        mlab.points3d(self.cargo_pos.data[0, 0], self.cargo_pos.data[0, 1], obj_h.data[0, 2] + 25, 50, scale_factor=1, color=(1, 1, 1))
        mlab.show()
