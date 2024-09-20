from setuptools import find_packages, setup

setup(
    name='deep_eit',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'resist=reconstruction.resist:resist',
        ],
    },
)