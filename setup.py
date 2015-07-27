from setuptools import setup, find_packages
setup(name='MultitaskDescriptor',
      version='1.0',
      url='https://github.com/vmolina/MultitaskDescriptor'
      description='A Python implementation of Multitas Descriptro',
      author='Victor Bellon',
      author_email='victor.bellon@mines-paristech.fr',
      packages=find_packages(), 
      requires=['numpy', 'sklearn', 'hyperopt', ],
      license='GNU GENERAL PUBLIC Version 2'
      )
