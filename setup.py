from setuptools import setup

setup(name='perisim',
      version='0.1',
      description='A package for simulating peristaltic tables',
      url='https://github.com/stokesresearchgroup/perisim',
      author='Ross M. McKenzie',
      author_email='r.m.mckenzie@ed.ac.uk',
      license='MIT',
      packages=['perisim'],
      install_requires=[
          'pytorch',
      ],
      zip_safe=False)