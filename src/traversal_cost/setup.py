from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Traversal cost Python package'

# Setting up
setup(
       # The name must match the folder name 'traversal_cost'
        name="traversalcost", 
        version=VERSION,
        author="Tom Ravaud",
        author_email="<tom.ravaud@eleves.enpc.fr>",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # Add any additional packages that 
        # needs to be installed along with your package.
        
        keywords=['python', 'traversal cost'],
        classifiers= [
            "Development Status :: 1 - Planning",
            "License :: OSI Approved :: MIT License",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            'Operating System :: POSIX :: Linux',
        ]
)
