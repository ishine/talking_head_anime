from utils.logging.tensorboard import tensorboardLogger


def get_logger(conf):
    return tensorboardLogger(conf)
