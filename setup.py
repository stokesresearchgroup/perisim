from setuptools import setup

setup(name='perisim',
      version='0.45',
      description='A package for simulating peristaltic tables',
      long_description='''
      A peristaltic table simulation.

      This package simulates a square grid of peristaltic cells beneath a flexible surface. Each cell is modelled as a gaussian disturbance in the flexible surface. Each cell can actuate to increase or decrease its amplitude. Objects on the surface then roll down the gradients of the surface.

      The simulation can randomly vary its parameters in order to allow for controller optimization using the radical envelope-of-noise hypothesis [Evolutionary Robotics and the Radical Envelope-of-Noise Hypothesis, Nick Jakobi, 1997].
      ''',
      url='https://github.com/stokesresearchgroup/perisim',
      author='Ross M. McKenzie',
      author_email='r.m.mckenzie@ed.ac.uk',
      license='MIT',
      packages=['perisim'],
      install_requires=[
          'torch',
      ],
      zip_safe=False)
