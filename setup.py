from setuptools import setup

setup(
    name='jade',
    version='0.1.0',
    description='Just Another DEep learning framework',
    python_requires='>=3',
    url='https://github.com/mfederici/JADE',
    author='Marco Federici',
    author_email='m.federici@uva.nl',
    license='MIT',
    packages=['jade','jade/run_manager'],
    scripts=['bin/jade'],
    install_requires=['torch>=1.0',
                      'torchvision',
                      'torchaudio',
                      'numpy',
                      'envyaml',
                      'future',
                      'python-dotenv',
                      'tqdm',
                      'matplotlib',
                      'wandb'],

    classifiers=[
        'Programming Language :: Python :: 3.4',
    ],
)