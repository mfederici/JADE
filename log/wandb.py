from log.base import LogWriter
import wandb


class WandBLogWriter(LogWriter):
    def log(self, name, value, entry_type, iteration):
        if entry_type == 'scalar':
            wandb.log({name: value}, step=iteration)
        else:
            raise Exception('Type %s is not recognized by WandBLogWriter' % type)
