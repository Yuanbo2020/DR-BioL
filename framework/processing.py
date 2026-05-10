import time, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from framework.utilities import create_folder
from framework.models import move_data_to_gpu
import framework.config as config
from sklearn.metrics import accuracy_score
from framework.data_mixup import Mixup

def forward(model, generate_func, device, return_names=False):
    output = []
    label = []

    output_domain = []
    label_domain = []

    audio_names = []
    # Evaluate on mini-batch

    for num, data in enumerate(generate_func):
        (batch_x, batch_y_class, batch_y_domain) = data

        batch_x = move_data_to_gpu(batch_x, device)

        model.eval()
        with torch.no_grad():
            output_linear_class, output_linear_domain, event_embeddings, location_embeddings = model(batch_x)

            output.append(output_linear_class.data.cpu().numpy())
            # ------------------------- labels -------------------------------------------------------------------------
            label.append(batch_y_class)

            output_domain.append(output_linear_domain.data.cpu().numpy())
            label_domain.append(batch_y_domain)

    dict = {}

    if return_names:
        dict['audio_names'] = np.concatenate(audio_names, axis=0)

    dict['prediction'] = np.concatenate(output, axis=0)
    # ----------------------------- labels -------------------------------------------------------------------------
    dict['label'] = np.concatenate(label, axis=0)

    dict['prediction_domain'] = np.concatenate(output_domain, axis=0)
    dict['label_domain'] = np.concatenate(label_domain, axis=0)
    return dict


def evaluate(model, generate_func, device):
    # Forward
    dict = forward(model=model, generate_func=generate_func, device=device)

    # MSC
    val_acc = accuracy_score(dict['label'], np.argmax(dict['prediction'], axis=1))

    val_acc_domain = accuracy_score(dict['label_domain'], np.argmax(dict['prediction_domain'], axis=1))

    return val_acc, val_acc_domain


class SupervisedContrastiveLoss(nn.Module):
    """
    Implemented supervised contrast loss and added an `invert` parameter to handle special Domain contrastive Loss.
    """

    def __init__(self, temperature=0.1, device='cpu'):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, embeddings, labels, invert=False):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        if invert:
            positives_mask = ~labels_matrix
        else:
            positives_mask = labels_matrix

        mask_no_diag = ~torch.eye(labels.shape[0], dtype=torch.bool, device=self.device)
        positives_mask = positives_mask * mask_no_diag

        if positives_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        logits = similarity_matrix / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        denominator_logits = logits.masked_fill(~mask_no_diag, float('-inf'))
        log_prob_denominator = torch.logsumexp(denominator_logits, dim=1)

        positive_logits = logits.masked_fill(~positives_mask, float('-inf'))
        log_prob_numerator = torch.logsumexp(positive_logits, dim=1)

        loss = log_prob_denominator - log_prob_numerator
        valid_samples_mask = positives_mask.sum(dim=1) > 0
        loss = loss[valid_samples_mask].mean()

        return loss


# MMD
def _pairwise_sq_dists(x, y):
    x2 = (x**2).sum(1).unsqueeze(1)        # [m,1]
    y2 = (y**2).sum(1).unsqueeze(0)        # [1,n]
    d2 = x2 + y2 - 2.0 * (x @ y.t())       # [m,n]
    return d2.clamp_min_(0)

def _gaussian_kernel(x, y, sigmas):
    if isinstance(sigmas, (float, int)):
        sigmas = [float(sigmas)]
    d2 = _pairwise_sq_dists(x, y)          # [m,n]
    K = 0.0
    for s in sigmas:
        K = K + torch.exp(-d2 / (2.0 * s * s))
    return K

def mmd2_unbiased(x, y, sigmas):
    m, n = x.size(0), y.size(0)
    Kxx = _gaussian_kernel(x, x, sigmas)
    Kyy = _gaussian_kernel(y, y, sigmas)
    Kxy = _gaussian_kernel(x, y, sigmas)
    # 去对角的无偏估计
    sum_Kxx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
    sum_Kyy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
    sum_Kxy = Kxy.mean()

    loss = sum_Kxx + sum_Kyy - 2.0 * sum_Kxy
    return torch.clamp(loss, min=0.).float()  # 最后双重保险：非负化

@torch.no_grad()
def _median_heuristic_sigma(z, max_samples=2048):
    # 估计一个合理的核带宽：全局中位数距离
    if z.size(0) > max_samples:
        idx = torch.randperm(z.size(0), device=z.device)[:max_samples]
        z = z[idx]
    d2 = _pairwise_sq_dists(z, z)
    tri = d2.triu(1)
    med = tri[tri > 0].median()
    s = torch.sqrt(med + 1e-12).item() if med.numel() > 0 else 1.0
    return s

def class_conditional_mmd_species_only(z: torch.Tensor, species: torch.Tensor, repeats: int = 4, min_per_group: int = 4, sigmas=None):
    z = z.double()
    device = z.device
    if sigmas is None:
        s = _median_heuristic_sigma(z)
        sigmas = [max(s/2, 1e-3), max(s, 1e-3), max(2*s, 1e-3), max(4*s, 1e-3)]
    loss, count = 0.0, 0
    for y in species.unique():
        idx = (species == y).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n < 2*min_per_group:
            continue

        Zy = z[idx]
        for _ in range(repeats):
            perm = torch.randperm(n, device=device)
            m = n // 2
            A = Zy[perm[:m]]
            B = Zy[perm[m:2*m]]
            if A.size(0) >= min_per_group and B.size(0) >= min_per_group:
                mmd_loss = mmd2_unbiased(A, B, sigmas)
                if torch.any(torch.isnan(mmd_loss)):
                    print('mmd_loss: ', mmd_loss, sigmas)
                    print('species: ', species)
                    print('species.unique(): ', species.unique())
                    print('y: ', y)
                    print('idx: ', idx)
                else:
                    loss = loss + mmd_loss
                    count += 1
    if count == 0:
        return z.new_tensor(0.0, dtype=z.dtype)
    loss = loss / count
    return torch.clamp(loss, min=0.).float() # 最后双重保险：非负化


def training_Dr_BioL(generator, device, model, models_dir, epochs, batch_size, num_classes, alpha=None, cs_temperature=0.01,
                                        lr_init=1e-3, log_path=None):
    mixup_fn = Mixup( mixup_alpha=0.8, cutmix_alpha=0, cutmix_minmax=None, prob=0.9, switch_prob=0,
        mode='batch', label_smoothing=0, num_classes=num_classes, )
    create_folder(models_dir)

    criterion_cc = SupervisedContrastiveLoss(temperature=cs_temperature, device=device)
    criterion_dc = SupervisedContrastiveLoss(temperature=cs_temperature, device=device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr_init)

    if generator.class_weights is not None:
        class_weights = move_data_to_gpu(generator.class_weights, device)

    max_val_acc = -np.inf
    max_val_acc_itera = 0
    save_val_best = 0
    list_val_acc = []
    list_val_acc_file = os.path.join(log_path, 'val_acc.txt')

    list_val_domain_acc = []
    list_val_domain_acc_file = os.path.join(log_path, 'val_acc_domain.txt')

    max_training_acc = -np.inf
    max_training_acc_itera = 0
    save_training_best = 0
    list_training_acc = []
    list_training_acc_file = os.path.join(log_path, 'training_acc.txt')

    list_training_domain_acc = []
    list_training_domain_acc_file = os.path.join(log_path, 'training_acc_domain.txt')

    # ------------------------------------------------------------------------------------------------------------------

    sample_num = len(generator.y_train_domain)
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')

    training_start_time = time.time()
    overrun_counter = 0
    break_flag = False

    for iteration, all_data in enumerate(generator.generate_training()):
        Epoch = iteration / one_epoch

        (batch_x, batch_y_class, batch_y_domain) = all_data

        batch_x = move_data_to_gpu(batch_x, device)
        batch_y_class_cpu = batch_y_class
        batch_y_domain_cpu = batch_y_domain
        batch_y_class = move_data_to_gpu(batch_y_class, device)
        batch_y_domain = move_data_to_gpu(batch_y_domain, device)

        train_bgn_time = time.time()
        model.train()
        optimizer.zero_grad()

        batch_y_class_binary = F.one_hot(batch_y_class.long(), num_classes=num_classes).float()
        batch_x, batch_y_class_binary = mixup_fn(batch_x, batch_y_class_binary)

        batch_pre_linear_class, batch_pre_linear_domain, event_embeddings, domain_embeddings = model(batch_x)
        loss_fn = torch.nn.BCEWithLogitsLoss
        loss_class = loss_fn(reduction='mean', weight=class_weights)(batch_pre_linear_class, batch_y_class_binary)

        # domain loss
        domain_softmax = F.log_softmax(batch_pre_linear_domain, dim=-1)
        loss_domain = F.nll_loss(domain_softmax, batch_y_domain)

        # contrastive
        loss_class_contrastive = criterion_cc(event_embeddings, batch_y_class, invert=False)
        loss_domain_contrastive = criterion_dc(domain_embeddings, batch_y_domain, invert=True)

        # proto MMD
        mmd_loss = class_conditional_mmd_species_only(event_embeddings, batch_y_class, repeats=4)

        print(
              'loss_cl: %.6f' % float(loss_class),
              'loss_con_cl: %.6f' % float(loss_class_contrastive),
              'mmd_loss: %.6f' % float(mmd_loss),
              'loss_D: %.6f' % float(loss_domain),
              'loss_con_D: %.6f' % float(loss_domain_contrastive),
              )

        if alpha is not None:
            if type(alpha[0]) == str:
                alpha = [float(each) for each in alpha]
                loss = (alpha[0] * loss_class + alpha[1] * loss_class_contrastive + alpha[2] * mmd_loss
                        + alpha[3] * loss_domain + alpha[4] * loss_domain_contrastive)
            else:
                loss = (alpha[0] * loss_class + alpha[1] * loss_class_contrastive + alpha[2] * mmd_loss
                        + alpha[3] * loss_domain + alpha[4] * loss_domain_contrastive)
        else:
            loss = loss_class + loss_class_contrastive + mmd_loss + loss_domain + loss_domain_contrastive

        print('epoch: ', '%.3f' % (Epoch), 'loss: %.6f' % float(loss),
              'loss: %.6f' % float(loss),
              'loss_cl: %.6f' % float(loss_class),
              'loss_con_cl: %.6f' % float(loss_class_contrastive),
              'mmd_loss: %.6f' % float(mmd_loss),
              'loss_D: %.6f' % float(loss_domain),
              'loss_con_D: %.6f' % float(loss_domain_contrastive),
              )

        loss.backward()
        optimizer.step()

        batch_pre_linear_class = batch_pre_linear_class.data.cpu().numpy()
        train_acc = accuracy_score(batch_y_class_cpu, np.argmax(batch_pre_linear_class, axis=1))
        list_training_acc.append(train_acc)

        batch_pre_linear_domain = batch_pre_linear_domain.data.cpu().numpy()
        train_domain_acc = accuracy_score(batch_y_domain_cpu, np.argmax(batch_pre_linear_domain, axis=1))
        list_training_domain_acc.append(train_domain_acc)

        print('epoch: ', '%.3f' % (Epoch), 'loss: %.6f' % float(loss), 'Train_acc: %.6f' % float(train_acc),
              'Train_D: %.6f' % float(train_domain_acc),
              'loss: %.6f' % float(loss),
              'loss_cl: %.6f' % float(loss_class),
              'loss_con_cl: %.6f' % float(loss_class_contrastive),
              'mmd_loss: %.6f' % float(mmd_loss),
              'loss_D: %.6f' % float(loss_domain),
              'loss_con_D: %.6f' % float(loss_domain_contrastive),
              )

        if iteration % one_epoch == 0 and iteration > 1:
            train_fin_time = time.time()
            generate_func = generator.generate_validate(data_type='validate')
            val_acc, val_acc_domain = evaluate(model=model, generate_func=generate_func, device=device)
            list_val_acc.append(val_acc)
            list_val_domain_acc.append(val_acc_domain)

            val_time = time.time() - train_fin_time

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                save_val_best = 1
                max_val_acc_itera = Epoch

            if train_acc > max_training_acc:
                max_training_acc = train_acc
                save_training_best = 1

                if Epoch >= config.warmup_epoch:
                    overrun_counter = -1

            print('E: ', '%.3f' % (Epoch), 'val_acc: %.3f' % float(val_acc), 'val_D_acc: %.3f' % float(val_acc_domain),)

            print('E: {}, T_val: {:.3f} s, max_val_acc: {:.3f} , itera: {} '
                  .format('%.4f' % (Epoch), val_time, max_val_acc, max_val_acc_itera))

            np.savetxt(list_val_acc_file, list_val_acc, fmt='%.5f')
            np.savetxt(list_training_acc_file, list_training_acc, fmt='%.5f')
            np.savetxt(list_val_domain_acc_file, list_val_domain_acc, fmt='%.5f')
            np.savetxt(list_training_domain_acc_file, list_training_domain_acc, fmt='%.5f')

            if save_val_best:
                save_val_best = 0
                save_out_dict = model.state_dict()
                save_out_path = os.path.join(models_dir, 'best_val_acc' + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Best val model saved to {}'.format(save_out_path))

            if save_training_best:
                save_training_best = 0
                save_out_dict = model.state_dict()
                save_out_path = os.path.join(models_dir, 'best_training_acc' + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Best val model saved to {}'.format(save_out_path))

            if Epoch >= config.warmup_epoch:
                overrun_counter += 1
                print(
                    'Epoch: %d, Train Acc: %.8f, Val Acc: %.8f, overrun_counter %i' % (
                        Epoch, train_acc, val_acc, overrun_counter))

            train_time = train_fin_time - train_bgn_time
            val_end_time = time.time()
            validate_time = val_end_time - train_fin_time
            print('epoch: {}, train time: {:.3f} s, iteration time: {:.3f} ms, validate time: {:.3f} s, '
                  'inference time : {:.3f} ms'.format('%.2f' % (Epoch), train_time,
                                                      (train_time / sample_num) * 1000, validate_time,
                                                      1000 * validate_time / sample_num))

        if Epoch >= config.warmup_epoch:
            if overrun_counter > config.early_stopping_max_overrun:
                break_flag = True

        if iteration > (epochs * one_epoch):
            break_flag = True

        if  break_flag:
            finish_time = time.time() - training_start_time
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print("All epochs are done.")

            save_out_dict = model.state_dict()
            save_out_path = os.path.join(models_dir, 'final_model' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Final model saved to {}'.format(save_out_path))

            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print('Model training finish time: {:.3f} s,'.format(finish_time))

            print('Training is done!!!')
            break





