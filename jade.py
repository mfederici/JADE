#!/usr/bin/env python

import os
import yaml
from dotenv import dotenv_values
import click
from jade.run_manager.wandb import WANDBRunManager

# TODO implement way to use different code from the one store
# TODO implement load checkpoint

@click.group(name='jade')
def jade():
    pass

@jade.command()
@click.argument('config-files',
              type=click.File('r'),
              nargs=-1,
             # help='List of configuration .yml files containing the description of "model", "data", "trainer" '
             #      'and "evaluators',
              required=True
              )
@click.option('--train-for',
              type=click.STRING,
              help='Total number of training time in epochs, iterations, seconds, minutes, hours, days or custom.'
                   ' (e.g. \"10 minutes\", \"10000 iterations\")',
              required=True
              )
@click.option('--code-dir',
              type=click.Path(file_okay=False),
              help='path to the folder containing the code (it must contain the "architectures", "data", "eval"'
                   ', "models" (and "trainers") sub-directories.'
              )
@click.option('--run_name',
              type=click.STRING,
              help='Name of the run')
@click.option('--verbose',
              type=click.BOOL,
              default=False,
              help='Verbose output')
@click.option('--device',
              type=click.STRING,
              default=False,
              help='Device on which the experiments are run ("cpu", "cuda", ...)'
              )
@click.option('--env-file',
              type=click.File('r'),
              default=open('.env','r'),
              help='Root directory in which the experiments are saved/loaded')
@click.option('--experiments_root',
              type=click.Path(file_okay=False),
              help='Root directory in which the experiments are saved/loaded')
def train(train_for, config_files, device='cpu', verbose=False, env_file='.env', overwrite='', experiments_root=None, code_dir='code', run_name=None):
    code_dir = code_dir.strip('"').strip('\'')
    env_file = env_file.name.strip('"').strip('\'')
    if not(run_name is None):
        run_name = run_name.strip('"').strip('\'')
    train_for = train_for.strip('"').strip('\'')

    update_env(env_file)

    config = read_and_merge(config_files, verbose=verbose)
    update_config(config, overwrite)

    if 'DEVICE' in os.environ:
        device = os.environ['DEVICE']

    run_manager = WANDBRunManager(run_name=run_name, config=config,
                                  wandb_dir=experiments_root,
                                  code_dir=code_dir,
                                  verbose=verbose)

    run_manager.run(train_amount=train_for, device=device)

@jade.command()
@click.argument('run-id',
                type=click.STRING,
                required=True)
@click.option('--train-for',
              type=click.STRING,
              help='Total number of training time in epochs, iterations, seconds, minutes, hours, days or custom.'
                   ' (e.g. \"10 minutes\", \"10000 iterations\")',
              required=True
              )
@click.option('--verbose',
              type=click.BOOL,
              default=False,
              help='Verbose output')
@click.option('--device',
              type=click.STRING,
              default=False,
              help='Device on which the experiments are run ("cpu", "cuda", ...)'
              )
@click.option('--env-file',
              type=click.File('r'),
              default=open('.env','r'),
              help='Root directory in which the experiments are saved/loaded')
@click.option('--experiments_root',
              type=click.Path(file_okay=False),
              help='Root directory in which the experiments are saved/loaded')
def resume(run_id, train_for, verbose=False, env_file='.env', experiments_root=None, device='cpu'):
    update_env(env_file)

    if 'DEVICE' in os.environ:
        device = os.environ['DEVICE']

    run_manager = WANDBRunManager(run_id=run_id,
                                  wandb_dir=experiments_root,
                                  verbose=verbose)

    run_manager.run(train_amount=train_for, device=device)


def update_env(env_file, verbose=False):
    env = dotenv_values(env_file)
    for k, v in env.items():
        os.environ[k] = v
        if verbose:
            print('Setting %s=%s' % (k, str(v)))


def read_and_merge(config_files, verbose=False):
    config = {}
    if len(config_files) > 0:
        if len(config_files) > 0:
            for config_file in config_files:
                filename = config_file.name
                with open(filename, 'r') as f:
                    d = yaml.safe_load(f)
                for k in d:
                    if '.' in d:
                        raise Exception('The special character \'.\' can not be used in the definition of %s' % k)
                    if k in config:
                        print('Warning: Duplicate entry for %s, the value %s is overwritten by %s' % (
                        k, str(config[k]), (d[k])))
                    else:
                        config[k] = d[k]
    return config


def update_config(config, overwrite, verbose=False):
     # parse and update the configuration from the override argument
     print('overwrite: %s' % str(overwrite))
     for entry in overwrite:
         key, value = entry.split('=')[0], entry.split('=')[1]
         value = yaml.safe_load(value)

         d = config
         last_d = d
         for sub_key in key.split('.'):
             if not (sub_key in d):
                 raise Exception(
                     'The parameter %s in %s specified in the --overwrite flag is not defined in the configuration.\n'
                     'The accessible keys are %s' %
                     (sub_key, key, str(d.keys()))
                 )
             last_d = d
             d = d[sub_key]

         last_d[key.split('.')[-1]] = value

         if verbose:
             print('%s\n\tOriginal value: %s\n\tOverwritten value: %s' % (key, str(d), str(value)))
     return config


if __name__ == '__main__':
    jade()