from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Params Python package'

# Setting up
setup(
       # The name must match the folder name 'params'
        name="params", 
        version=VERSION,
        author="Tom Ravaud",
        author_email="<tom.ravaud@eleves.enpc.fr>",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # Add any additional packages that 
        # needs to be installed along with your package.
        
        keywords=['python', 'params'],
        classifiers= [
            "Development Status :: 1 - Planning",
            "License :: OSI Approved :: MIT License",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            'Operating System :: POSIX :: Linux',
        ]
)

# To install this package, run "pip install ." in the terminal
