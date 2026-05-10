import sys, os, argparse, torch, pickle
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from framework.data_generator import *
from framework.models import *
from framework.utilities import create_folder
import framework.config as config
import numpy as np
from sklearn.metrics import accuracy_score

# gpu_id = 1
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


import sklearn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve
from matplotlib import pyplot

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def plot_ROC_AUC_multiple_class(classes, y_true, y_pred_prob, plot_dir=None, filename=None, show=False,
                                plot_and_save=False, replot=False):
    # calculate AUC

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    threshold = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], threshold[i] = sklearn.metrics.roc_curve(to_categorical(y_true)[:, i], y_pred_prob[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    # # 由于prediction只有0，1两类，相当于绘图中只有3个点的数值
    # print('fpr[i]: ', fpr)
    # print('tpr[i]: ', tpr)
    # print('threshold[i]: ', threshold)
    # # fpr[i]:  {0: array([0. , 0.5, 1. ]), 1: array([0.  , 0.12, 1.  ])}
    # # tpr[i]:  {0: array([0.  , 0.88, 1.  ]), 1: array([0. , 0.5, 1. ])}
    # # threshold[i]:  {0: array([inf,  1.,  0.]), 1: array([inf,  1.,  0.])}
    # # fpr[micro]:  [0.         0.18333333 1.        ]
    # # tpr[micro]:  [0.         0.81666667 1.        ]
    # # threshold[micro]:  [inf  1.  0.]

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], threshold["micro"] = sklearn.metrics.roc_curve(to_categorical(y_true).ravel(),
                                                                               y_pred_prob.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

    # print('fpr[micro]: ', fpr["micro"])
    # print('tpr[micro]: ', tpr["micro"])
    # print('threshold[micro]: ', threshold["micro"] )

    if plot_and_save:
        create_folder(plot_dir)
        with open(os.path.join(plot_dir, filename + '_roc_micro_' + '%.4f' % roc_auc["micro"] + '.txt'),
                  "w") as text_file:
            print(roc_auc, file=text_file)

    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    if plot_and_save:
        png_file = os.path.join(plot_dir, filename + '_ROC_AUC_' + '%.4f' % roc_auc["micro"] + '.pdf')

        if not os.path.exists(png_file) or replot:
            plt.clf()
            plt.close()
            plt.figure(figsize=(8, 8))
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.3f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.3f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i in range(len(classes)):
                plt.plot(fpr[i], tpr[i], lw=lw,
                         label='{0} (area = {1:0.3f})'
                               ''.format(classes[i], roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title('ROC AUC micro: %.4f , macro: %.4f ' % (roc_auc["micro"], roc_auc["macro"]))
            plt.xlabel('False positives rate(FPR) / (1-Specificity)')
            plt.ylabel('True positives rate(TPR) / Sensitivity / Recall')
            plt.legend(loc="lower right")

            # show the plot
            plt.tight_layout()
            # plt.grid(True)
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.savefig(png_file)
            if show:
                plt.show()
            plt.close()
            plt.clf()
    return roc_auc


def plot_PR_curve_AUC_multiple_class(classes, y_true, y_pred_prob, plot_dir, filename, show=False, replot=False):
    precision = dict()
    recall = dict()
    threshold = dict()
    average_precision = dict()
    Y_test = to_categorical(y_true)
    y_score = y_pred_prob

    n_classes = len(classes)

    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(Y_test[:, i],
                                                                       y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # # 由于prediction只有0，1两类，相当于绘图中只有2个点的数值
    # print('precision[i]: ', precision)
    # print('recall[i]: ', recall)
    # print('threshold[i]: ', threshold)
    # precision[i]:  {0: array([0.83333333, 0.89795918, 1.        ]), 1: array([0.16666667, 0.45454545, 1.        ])}
    # recall[i]:  {0: array([1.  , 0.88, 0.  ]), 1: array([1. , 0.5, 0. ])}
    # threshold[i]:  {0: array([0, 1]), 1: array([0, 1])}
    # precision[micro]:  [0.5        0.81666667 1.        ]
    # recall[micro]:  [1.         0.81666667 0.        ]
    # threshold[micro]:  [0 1]

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], threshold["micro"] = precision_recall_curve(Y_test.ravel(),
                                                                                     y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")

    # print('precision[micro]: ', precision["micro"])
    # print('recall[micro]: ', recall["micro"])
    # print('threshold[micro]: ', threshold["micro"])

    create_folder(plot_dir)
    with open(os.path.join(plot_dir, filename + '_pr_' + '%.4f' % average_precision["micro"] + '.txt'),
              "w") as text_file:
        print(average_precision, file=text_file)

    png_file = os.path.join(plot_dir, filename + '_MSC_PR_' + '%.4f' % average_precision["micro"] + '.pdf')
    if replot or not os.path.exists(png_file):
        plt.clf()
        plt.close()

        plt.figure(figsize=(8, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average (area = {0:0.3f})'
                      ''.format(average_precision["micro"]))

        for i in range(n_classes):
            l, = plt.plot(recall[i], precision[i], lw=2)
            lines.append(l)
            labels.append('{0} (area = {1:0.3f})'
                          ''.format(classes[i], average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average Precision / PR_Curve_AUC micro: %.4f' % (average_precision["micro"]))
        plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(png_file, bbox_inches='tight')

        if show:
            pyplot.show()
        plt.close()
        plt.clf()
    return average_precision


from matplotlib import pyplot as plt


def plot_confusion_matrix(cm, classes, filename=None,
                          normalize=False, cmap=plt.cm.Blues, figsize=(8, 8), replot=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    #     std = std * 100
    if normalize:
        cm_list = [cm]
        cm = []
        for item in cm_list:
            cm.append(item.astype('float') / item.sum(axis=1)[:, np.newaxis] * 100)
        cm = np.mean(cm, axis=0)  # Convert mean to normalised percentage
        # std = np.std(cm, axis=0)  # Standard deviation also in percentage

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        # std = std.astype('float') / std.sum(axis=1)[:, np.newaxis] *100
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, as input by user')

    # print(cm)

    if not os.path.exists(filename) or replot:
        plt.clf()
        plt.close()
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        #     ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,

               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else '.2f'
        fmt_std = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # ax.text(j, i, format(cm[i, j], fmt) + '±' + format(std[i, j], fmt_std),
                #         ha="center", va="center",
                #         color="white" if cm[i, j] > thresh else "black")
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(filename)
        plt.clf()
        plt.close()
    return cm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


import shutil
def copy_save_good_dir(metric, save_good_dir):
    dir_name = metric[1]
    source = os.path.join(os.getcwd(), dir_name)
    print(os.path.dirname(__file__))

    target_dir = os.path.join(save_good_dir, dir_name)
    if not os.path.exists(target_dir):
        print('copy:', source, ' ------> ', target_dir)
        shutil.copytree(source, target_dir)


def main(argv):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    batch_size = 64
    generator = DataGenerator_4domains_8species(batch_size)

    num_classes = len(config.all_species_labels)
    using_classes = config.all_species_labels

    workspace = os.path.join(os.getcwd())

    model_name = 'DR_BioL_CNN'
    # model_name = 'DAT_CNN'
    model_path = os.path.join(workspace, 'Pretrained_models', model_name + '.pth')

    using_model = eval(model_name)

    model = using_model(class_num=num_classes, domain_num=len(config.all_domain_labels), dropout=0.2,
                        MC_dropout=True, batchnormal=True)
    # model = using_model(class_num=num_classes, domain_num=len(config.all_domain_labels), dropout=0.2,
    #                         MC_dropout=False, batchnormal=True)

    pretrained_model = torch.load(model_path, map_location=device)
    model.load_state_dict(pretrained_model, strict=True)
    model.to(device)

    generate_func = generator.generate_testing(batch_size)
    all_batch_y = []
    all_pred_y = []
    for num, data in enumerate(generate_func):
        (batch_x, batch_y_class,_) = data
        batch_x = move_data_to_gpu(batch_x, device)
        model.eval()
        with torch.no_grad():
            y_pred_linear, output_linear_domain, event_embeddings, location_embeddings = model(batch_x)
            all_batch_y.append(batch_y_class)
            all_pred_y.append(y_pred_linear.data.cpu().numpy())

    all_pred_y = np.concatenate(all_pred_y)
    all_batch_y = np.concatenate(all_batch_y)
    print(all_batch_y.shape, all_pred_y.shape)
    test_acc = accuracy_score(all_batch_y, np.argmax(all_pred_y, axis=1))
    print('MSC acc:', test_acc)
    # (10256,) (10256, 8)
    # MSC acc: 0.8565717628705148
    # MSC acc: 0.796899375975039

    y_true_label = all_batch_y
    y_pred_hard = np.argmax(all_pred_y, axis=1)
    y_pred_soft = softmax(all_pred_y)

    cm = confusion_matrix(y_true_label, y_pred_hard)
    plot_dir = os.path.join(workspace, 'Inference')
    create_folder(plot_dir)
    figsize = (18, 18)
    output_confusion_file = os.path.join(plot_dir, model_name + '_cm.png')
    plot_confusion_matrix(cm, using_classes, output_confusion_file,
                          normalize=False, figsize=figsize)

    output_confusion_file = os.path.join(plot_dir, model_name + '_cm_normalize.png')
    plot_confusion_matrix(cm, using_classes, output_confusion_file,
                          normalize=True, figsize=figsize)

    # PR
    using_predictions = y_pred_soft

    roc_auc = plot_ROC_AUC_multiple_class(using_classes, y_true_label,
                                          using_predictions, filename=model_name,
                                          plot_dir=plot_dir, show=False,
                                          plot_and_save=True)

    average_precision = plot_PR_curve_AUC_multiple_class(using_classes, y_true_label,
                                                         using_predictions,
                                                         filename=model_name,
                                                         plot_dir=plot_dir)



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)






