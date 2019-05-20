# perisim
### Installation

`pip install perisim`

### Usage

Make a perisim object with:

```
sim = PeriSim(x, y, [start])
```

x, y, cargoPos, amplitude = 5., spacing = 80, stdDev = 40,
                 timeStep = 0.01, variance = 0.2, cargoVel = None, height = 1.,
                 cargoWeight = None, g = 9800, friction = 0.01, actForce = 100,
                 actTime = 0.1, gpu = False