from log.base import LogWriter
import wandb


class WandBLogWriter(LogWriter):
    def __init__(self, experiment_dir, data_config, trainer_config, eval_config):
        config = {
            'trainer': trainer_config,
            'data': data_config,
            'eval': eval_config
        }

        wandb.init(project="causal-mib", name=experiment_dir, config=config, dir=experiment_dir)

    def log(self, name, value, entry_type, iteration):
        if entry_type == 'scalar':
            wandb.log({name: value}, step=iteration)
        else:
            raise Exception('Type %s is not recognized by WandBLogWriter' % type)