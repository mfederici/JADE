from log.base import LogWriter


class TensorboardLogWriter(LogWriter):
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, name, value, entry_type, iteration):
        if entry_type == 'scalar':
            self.writer.add_scalar(tag=name, scalar_value=value, global_step=iteration)
        else:
            raise Exception('Type %s is not recognized by TensorboardLogWriter' % type)
