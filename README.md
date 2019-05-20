# perisim
### Description
A peristaltic table simulation.

This package simulates a square grid of peristaltic cells beneath a flexible surface. 
Each cell is modelled as a gaussian disturbance in the flexible surface. Each cell can actuate to increase or decrease its amplitude. 
Objects on the surface then roll down the gradients of the surface.

The simulation can randomly vary its parameters in order to allow for controller optimization using the radical envelope-of-noise hypothesis 
[Evolutionary Robotics and the Radical Envelope-of-Noise Hypothesis, Nick Jakobi, 1997].

 Any units can be used as long as the parameters are updated to be consistent, the default units are millimeters, seconds and kilograms.

### Installation

`pip install perisim`

### Usage

Make a perisim object with:

```
sim = PeriSim(x, y, cargo_pos)
```

The required arguments are:

```
x: The number of peristaltic cell columns of the table

y: The number of peristaltic cell rows of the table

cargo_pos: A list of initial XY coordinates of the cargo objects moving on the table e.g [[1, 2], [2,3]].
```

The parameters of the simulation are Keyword arguments:
```
amplitude: The height which a cell can expand or contract by. Default 5.

spacing: The spacing of rows and columns. Default 80.
  
stddev: The standard deviation of the Gaussian disturbance of each cell. Default 40.

time_step: The time step of the simulation. Default 0.01.

variance: The maximum proportion that a parameter can be randomly varied by at startup. Default 0.

cargoVel: A list of initial XY coordinates of the cargo objects moving on the table. 
Needs to be the same length as cargo_pos. Assigns 0, 0 for all if None. Default None.

height: The rest height of the gaussian disturbance caused by a cell. Allows for surfaces that have slight deformations while the table is at rest. Default 0.

cargo_mass: A list of the masses of the cargo objects moving on the table. 
Needs to be the same length as cargo_pos. Assigns 0.01 for all if None. Default None.

g: Gravitational strength. Default 9800.

friction: The frictional force as a proportion of current velocity. Default 0.01.

act_force: The proportional increase in reaction force experienced when on an expanding cell. Default 100.

act_time: the time taken for a cell to actuate. Only effects actuation force, not gradients. Default 0.1.

gpu: Whether the simulation should use gpu acceleration. Only noticeable on very large tables. Default False.
```

To run the simulation for one time step:

```
sim.update()
```

To change the actuation of a cell with grid coordinates (x,y):
```
sim.actuate(x, y, direction)
```
where direction is:
```
0 for rest height
1 for extension
-1 for contraction
```