import sys, os, argparse, torch

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from framework.data_generator import *
from framework.models import *
from framework.processing import *
import framework.config as config


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr_rate', type=float, default=0.0005)
    parser.add_argument('-cs_temperature', type=float, default=0.01)
    parser.add_argument('-batch', type=int, default=64)
    parser.add_argument('-epochs', type=int, default=500)
    args = parser.parse_args()

    cs_temperature = args.cs_temperature
    lr_init = args.lr_rate
    batch_size = args.batch
    epochs = args.epochs
    model_name = 'DR_BioL_CNN'

    sys_name = ('sy_lr' + str(lr_init).replace('-', '') \
               + '_e' + str(epochs) + '_b' + str(batch_size) + '_w' + str(config.win_size) + '_'
                + str(cs_temperature) + '_wm' + str(config.warmup_epoch) + 'sp' + str(config.early_stopping_max_overrun))

    workspace = os.path.join(os.getcwd(), sys_name)

    create_folder(workspace)

    log_path = os.path.join(workspace, 'logs')
    create_folder(log_path)
    filename = os.path.basename(__file__).split('.py')[0]
    print_log_file = os.path.join(log_path, filename + '_print.log')
    sys.stdout = Logger(print_log_file, sys.stdout)
    console_log_file = os.path.join(log_path, filename + '_console.log')
    sys.stderr = Logger(console_log_file, sys.stderr)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    generator = DataGenerator_4domains_8species(batch_size)
    using_model = eval(model_name)

    num_classes = len(config.all_species_labels)
    print('num_classes: ', num_classes)

    model = using_model(class_num=num_classes, domain_num = len(config.all_domain_labels), dropout=0.2,
                        MC_dropout=True, batchnormal=True)
    model.to(device)

    models_dir = os.path.join(workspace, 'model')
    training_Dr_BioL(generator, device, model, models_dir, epochs, batch_size, num_classes, alpha=['1 1 1 1 1'],
                     cs_temperature=cs_temperature, lr_init=lr_init, log_path=log_path)

    print('Training is done!!!')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















