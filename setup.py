from setuptools import setup

setup(
    name='jade',
    version='0.1.0',
    description='Just Another DEep learning framework',
    url='https://github.com/mfederici/JADE',
    author='Marco Federici',
    author_email='m.federici@uva.nl',
    license='MIT',
    packages=['jade'],
    install_requires=['torch>=1.0',
                      'numpy',
                      'envyaml',
                      'python-dotenv',
                      'wandb'],

    classifiers=[
        'Programming Language :: Python :: 3.4',
    ],
)