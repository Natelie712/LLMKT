import numpy as np
import time
import random
import argparse
import pickle
import os
import gc
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from models import GKT, MultiHeadAttention, VAE, DKT
from metrics import KTLoss, VAELoss
from processing import load_dataset

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_false', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data-dir', type=str, default='data', help='Data dir for loading input data.')
parser.add_argument('--data-file', type=str, default='assistment_test15.csv', help='Name of input data file.')
parser.add_argument('--save-dir', type=str, default='logs', help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('-graph-save-dir', type=str, default='graphs', help='Dir for saving concept graphs.')
parser.add_argument('--load-dir', type=str, default='', help='Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
parser.add_argument('--dkt-graph-dir', type=str, default='dkt-graph', help='Where to load the pretrained dkt graph.')
parser.add_argument('--dkt-graph', type=str, default='dkt_graph.txt', help='DKT graph data file name.')
parser.add_argument('--model', type=str, default='GKT', help='Model type to use, support GKT and DKT.')
parser.add_argument('--hid-dim', type=int, default=32, help='Dimension of hidden knowledge states.')
parser.add_argument('--emb-dim', type=int, default=32, help='Dimension of concept embedding.')
parser.add_argument('--attn-dim', type=int, default=32, help='Dimension of multi-head attention layers.')
parser.add_argument('--vae-encoder-dim', type=int, default=32, help='Dimension of hidden layers in vae encoder.')
parser.add_argument('--vae-decoder-dim', type=int, default=32, help='Dimension of hidden layers in vae decoder.')
parser.add_argument('--edge-types', type=int, default=2, help='The number of edge types to infer.')
parser.add_argument('--graph-type', type=str, default='Dense', help='The type of latent concept graph.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--bias', type=bool, default=True, help='Whether to add bias for neural network layers.')
parser.add_argument('--binary', type=bool, default=True, help='Whether only use 0/1 for results.')
parser.add_argument('--result-type', type=int, default=12, help='Number of results types when multiple results are used.')
parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
parser.add_argument('--hard', action='store_true', default=False, help='Uses discrete samples in training forward pass.')
parser.add_argument('--no-factor', action='store_true', default=False, help='Disables factor graph model.')
parser.add_argument('--prior', action='store_true', default=False, help='Whether to use sparsity prior.')
parser.add_argument('--var', type=float, default=1, help='Output variance.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per batch.')
parser.add_argument('--train-ratio', type=float, default=0.6, help='The ratio of training samples in a dataset.')
parser.add_argument('--val-ratio', type=float, default=0.2, help='The ratio of validation samples in a dataset.')
parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset or not.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
parser.add_argument('--test', type=bool, default=False, help='Whether to test for existed model.')
parser.add_argument('--test-model-dir', type=str, default='logs/expDKT', help='Existed model file dir.')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

res_len = 2 if args.binary else args.result_type

# Save model and meta-data. Always saves in a new sub-folder.
log = None
save_dir = args.save_dir
if args.save_dir:
    exp_counter = 0
    now = datetime.datetime.now()
    # timestamp = now.isoformat()
    timestamp = now.strftime('%Y-%m-%d %H-%M-%S')
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    save_dir = '{}/exp{}/'.format(args.save_dir, model_file_name + timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    meta_file = os.path.join(save_dir, 'metadata.pkl')
    model_file = os.path.join(save_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    log_file = os.path.join(save_dir, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_dir provided!" + "Testing (within this script) will throw an error.")

# load dataset
dataset_path = os.path.join(args.data_dir, args.data_file)
dkt_graph_path = os.path.join(args.dkt_graph_dir, args.dkt_graph)
if not os.path.exists(dkt_graph_path):
    dkt_graph_path = None
concept_num, graph, train_loader, valid_loader, test_loader = load_dataset(dataset_path, args.batch_size, args.graph_type, dkt_graph_path=dkt_graph_path,
                                                                           train_ratio=args.train_ratio, val_ratio=args.val_ratio, shuffle=args.shuffle,
                                                                           model_type=args.model, use_cuda=args.cuda)

# build models
graph_model = None
if args.model == 'GKT':
    if args.graph_type == 'MHA':
        graph_model = MultiHeadAttention(args.edge_types, concept_num, args.emb_dim, args.attn_dim, dropout=args.dropout)
    elif args.graph_type == 'VAE':
        graph_model = VAE(args.emb_dim, args.vae_encoder_dim, args.edge_types, args.vae_decoder_dim, args.vae_decoder_dim, concept_num,
                          edge_type_num=args.edge_types, tau=args.temp, factor=args.factor, dropout=args.dropout, bias=args.bias)
        vae_loss = VAELoss(concept_num, edge_type_num=args.edge_types, prior=args.prior, var=args.var)
        if args.cuda:
            vae_loss = vae_loss.cuda()
    if args.cuda and args.graph_type in ['MHA', 'VAE']:
        graph_model = graph_model.cuda()
    model = GKT(concept_num, args.hid_dim, args.emb_dim, args.edge_types, args.graph_type, graph=graph, graph_model=graph_model,
                dropout=args.dropout, bias=args.bias, has_cuda=args.cuda)
elif args.model == 'DKT':
    model = DKT(res_len * concept_num, args.emb_dim, concept_num, dropout=args.dropout, bias=args.bias)
else:
    raise NotImplementedError(args.model + ' model is not implemented!')
kt_loss = KTLoss()

# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

# load model/optimizer/scheduler params
if args.load_dir:
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    model_file = os.path.join(args.load_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    model.load_state_dict(torch.load(model_file))
    optimizer.load_state_dict(torch.load(optimizer_file))
    scheduler.load_state_dict(torch.load(scheduler_file))
    args.save_dir = False

# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

if args.model == 'GKT' and args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    model = model.cuda()
    kt_loss = KTLoss()



TOTAL_ITEMS = 364  # total number of question slots in the assessment bank (L / Q for completion rate)


def _update_fairness_bins(bin_labels, bin_outputs, answers, pred_res, total_items=TOTAL_ITEMS):
    """Accumulate labels and prediction scores into completion-rate bins.

    Parameters
    ----------
    bin_labels : list[list[int]]
    bin_outputs : list[list[float]]
    answers : torch.Tensor, shape [B, L]
        Padded with -1.
    pred_res : torch.Tensor, shape [B, L-1]
        Predicted probabilities for next-step answers.
    total_items : int
        Q in completion_rate = L / Q.
    """
    if answers is None or pred_res is None:
        return
    if answers.dim() != 2 or pred_res.dim() != 2:
        return

    # real_answers used in KTLoss: timestamp 1..T
    real_answers = answers[:, 1:]
    mask = torch.ne(real_answers, -1)

    if mask.sum().item() == 0:
        return

    # L_i = number of valid answers per student (same as used in KTLoss)
    lengths = mask.sum(dim=1).cpu().numpy().astype(float)
    if total_items <= 0:
        total_items = 1

    probs = pred_res.detach().cpu().numpy()
    labels = real_answers.detach().cpu().numpy()
    mask_np = mask.cpu().numpy()

    B, T = labels.shape
    for i in range(B):
        valid = mask_np[i]
        if not valid.any():
            continue
        L_i = lengths[i]
        completion_rate = float(L_i) / float(total_items)
        # Map [0,1+] -> bins 0..9
        bin_idx = int(completion_rate * 10.0)
        if bin_idx < 0:
            bin_idx = 0
        if bin_idx > 9:
            bin_idx = 9

        y_true = labels[i][valid].astype(int)
        y_score = probs[i][valid].astype(float)

        bin_labels[bin_idx].extend(y_true.tolist())
        bin_outputs[bin_idx].extend(y_score.tolist())


def _compute_bin_stats(bin_labels, bin_outputs):
    """Compute TPR/FPR/ACC per non-empty completion bin.

    Returns
    -------
    stats : dict[int, dict[str, float]]
        stats[bin] = {'tpr': ..., 'fpr': ..., 'acc': ...}
    """
    stats = {}
    for b in range(10):
        y_true = np.asarray(bin_labels[b], dtype=int)
        y_score = np.asarray(bin_outputs[b], dtype=float)
        if y_true.size == 0:
            continue
        # binary predictions at 0.5 threshold
        y_pred = (y_score >= 0.5).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
        acc = float(tp + tn) / float(tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        stats[b] = {"tpr": tpr, "fpr": fpr, "acc": acc}
    return stats


def _compute_fairness_from_bins(bin_stats):
    """Compute Euclidean EO distance (lowest vs highest bin) and ACC variance."""
    if not bin_stats:
        return {"eo_dist": np.nan, "acc_var": np.nan, "non_empty_bins": []}

    non_empty = sorted(bin_stats.keys())
    low = non_empty[0]
    high = non_empty[-1]

    tpr_low = bin_stats[low]["tpr"]
    fpr_low = bin_stats[low]["fpr"]
    tpr_high = bin_stats[high]["tpr"]
    fpr_high = bin_stats[high]["fpr"]

    eo_dist = float(np.sqrt((tpr_low - tpr_high) ** 2 + (fpr_low - fpr_high) ** 2))
    accs = [bin_stats[b]["acc"] for b in non_empty]
    acc_var = float(np.var(accs)) if accs else np.nan

    return {"eo_dist": eo_dist, "acc_var": acc_var, "non_empty_bins": non_empty}


def _print_fairness_summary(bin_stats, fairness, prefix, log_handle=None):
    """Print fairness metrics in a clean block (console + optional log)."""
    lines = []
    lines.append(f"=== Fairness (by completion bins) [{prefix}] ===")
    non_empty = fairness.get("non_empty_bins", [])
    if non_empty:
        lines.append("Non-empty bins: " + ", ".join(str(b) for b in non_empty))
    else:
        lines.append("Non-empty bins: (none)")

    for b in sorted(bin_stats.keys()):
        s = bin_stats[b]
        lines.append(
            f"  Bin {b}: TPR={s['tpr']:.4f}, FPR={s['fpr']:.4f}, ACC={s['acc']:.4f}"
        )

    lines.append(
        f"EO dist (lowest vs highest non-empty bin): {fairness['eo_dist']:.6f}"
    )
    lines.append(
        f"Accuracy variance across bins: {fairness['acc_var']:.6f}"
    )

    for line in lines:
        print(line)
        if log_handle is not None:
            print(line, file=log_handle)
    if log_handle is not None:
        log_handle.flush()


def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []
    kt_train = []
    vae_train = []
    auc_train = []
    acc_train = []

    if graph_model is not None:
        graph_model.train()
    model.train()

    # --------------------
    # Training loop
    # --------------------
    for batch_idx, (features, questions, answers) in enumerate(train_loader):
        t1 = time.time()
        if args.cuda:
            features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()

        ec_list, rec_list, z_prob_list = None, None, None
        if args.model == 'GKT':
            pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
        elif args.model == 'DKT':
            pred_res = model(features, questions)
        else:
            raise NotImplementedError(args.model + ' model is not implemented!')

        loss_kt, auc, acc = kt_loss(pred_res, answers)
        kt_train.append(float(loss_kt.cpu().detach().numpy()))
        if auc != -1 and acc != -1:
            auc_train.append(auc)
            acc_train.append(acc)

        if args.model == 'GKT' and args.graph_type == 'VAE':
            if args.prior:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list, log_prior=log_prior)
            else:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
            vae_train.append(float(loss_vae.cpu().detach().numpy()))
            loss = loss_kt + loss_vae
        else:
            loss = loss_kt

        loss_train.append(float(loss.cpu().detach().numpy()))
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        del loss
        print('batch idx: ', batch_idx,
              'loss kt: ', loss_kt.item(),
              'auc: ', auc, 'acc: ', acc,
              'cost time: ', str(time.time() - t1))

    # --------------------
    # Validation loop (+ fairness)
    # --------------------
    loss_val = []
    kt_val = []
    vae_val = []
    auc_val = []
    acc_val = []

    bin_labels = [[] for _ in range(10)]
    bin_outputs = [[] for _ in range(10)]

    if graph_model is not None:
        graph_model.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, (features, questions, answers) in enumerate(valid_loader):
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()

            ec_list, rec_list, z_prob_list = None, None, None
            if args.model == 'GKT':
                pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')

            loss_kt, auc, acc = kt_loss(pred_res, answers)
            loss_kt_val = float(loss_kt.cpu().detach().numpy())
            kt_val.append(loss_kt_val)
            if auc != -1 and acc != -1:
                auc_val.append(auc)
                acc_val.append(acc)

            cur_loss = loss_kt_val
            if args.model == 'GKT' and args.graph_type == 'VAE':
                if args.prior:
                    loss_vae = vae_loss(ec_list, rec_list, z_prob_list, log_prior=log_prior)
                else:
                    loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                loss_vae_val = float(loss_vae.cpu().detach().numpy())
                vae_val.append(loss_vae_val)
                cur_loss = loss_kt_val + loss_vae_val
            loss_val.append(cur_loss)

            # update fairness bins
            _update_fairness_bins(bin_labels, bin_outputs, answers, pred_res, total_items=TOTAL_ITEMS)

    # --------------------
    # Logging
    # --------------------
    mean_loss_train = float(np.mean(loss_train)) if loss_train else 0.0
    mean_auc_train = float(np.mean(auc_train)) if auc_train else -1.0
    mean_acc_train = float(np.mean(acc_train)) if acc_train else -1.0

    mean_loss_val = float(np.mean(loss_val)) if loss_val else 0.0
    mean_auc_val = float(np.mean(auc_val)) if auc_val else -1.0
    mean_acc_val = float(np.mean(acc_val)) if acc_val else -1.0

    if args.model == 'GKT' and args.graph_type == 'VAE':
        mean_kt_train = float(np.mean(kt_train)) if kt_train else 0.0
        mean_vae_train = float(np.mean(vae_train)) if vae_train else 0.0
        mean_kt_val = float(np.mean(kt_val)) if kt_val else 0.0
        mean_vae_val = float(np.mean(vae_val)) if vae_val else 0.0

        msg = (
            f"Epoch: {epoch:04d} "
            f"loss_train: {mean_loss_train:.10f} kt_train: {mean_kt_train:.10f} vae_train: {mean_vae_train:.10f} "
            f"auc_train: {mean_auc_train:.10f} acc_train: {mean_acc_train:.10f} "
            f"loss_val: {mean_loss_val:.10f} kt_val: {mean_kt_val:.10f} vae_val: {mean_vae_val:.10f} "
            f"auc_val: {mean_auc_val:.10f} acc_val: {mean_acc_val:.10f} "
            f"time: {time.time() - t:.4f}s"
        )
    else:
        msg = (
            f"Epoch: {epoch:04d} "
            f"loss_train: {mean_loss_train:.10f} "
            f"auc_train: {mean_auc_train:.10f} acc_train: {mean_acc_train:.10f} "
            f"loss_val: {mean_loss_val:.10f} "
            f"auc_val: {mean_auc_val:.10f} acc_val: {mean_acc_val:.10f} "
            f"time: {time.time() - t:.4f}s"
        )

    print(msg)
    if args.save_dir and log is not None:
        print(msg, file=log)

    # Fairness summary for validation
    bin_stats = _compute_bin_stats(bin_labels, bin_outputs)
    fairness = _compute_fairness_from_bins(bin_stats)
    _print_fairness_summary(bin_stats, fairness, prefix="Validation", log_handle=log if args.save_dir else None)

    res = mean_loss_val

    # Save best model by validation loss
    if args.save_dir and res < best_val_loss:
        print('Best model so far, saving...')
        if log is not None:
            print('Best model so far, saving...', file=log)
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        torch.save(scheduler.state_dict(), scheduler_file)

    # cleanup
    del loss_train
    del auc_train
    del acc_train
    del loss_val
    del auc_val
    del acc_val
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()
    return res


def test():
    loss_test = []
    kt_test = []
    vae_test = []
    auc_test = []
    acc_test = []

    bin_labels = [[] for _ in range(10)]
    bin_outputs = [[] for _ in range(10)]

    if graph_model is not None:
        graph_model.eval()
    model.eval()
    model.load_state_dict(torch.load(model_file))
    with torch.no_grad():
        for batch_idx, (features, questions, answers) in enumerate(test_loader):
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            ec_list, rec_list, z_prob_list = None, None, None
            if args.model == 'GKT':
                pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')

            loss_kt, auc, acc = kt_loss(pred_res, answers)
            kt_test.append(float(loss_kt.cpu().detach().numpy()))
            if auc != -1 and acc != -1:
                auc_test.append(auc)
                acc_test.append(acc)

            cur_loss = float(loss_kt.cpu().detach().numpy())
            if args.model == 'GKT' and args.graph_type == 'VAE':
                if args.prior:
                    loss_vae = vae_loss(ec_list, rec_list, z_prob_list, log_prior=log_prior)
                else:
                    loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                vae_test.append(float(loss_vae.cpu().detach().numpy()))
                cur_loss = cur_loss + float(loss_vae.cpu().detach().numpy())
            loss_test.append(cur_loss)

            # accumulate fairness bins
            _update_fairness_bins(bin_labels, bin_outputs, answers, pred_res, total_items=TOTAL_ITEMS)

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    if args.model == 'GKT' and args.graph_type == 'VAE':
        msg = (
            f"loss_test: {np.mean(loss_test):.10f} "
            f"kt_test: {np.mean(kt_test):.10f} "
            f"vae_test: {np.mean(vae_test):.10f} "
            f"auc_test: {np.mean(auc_test):.10f} "
            f"acc_test: {np.mean(acc_test):.10f}"
        )
    else:
        msg = (
            f"loss_test: {np.mean(loss_test):.10f} "
            f"auc_test: {np.mean(auc_test):.10f} "
            f"acc_test: {np.mean(acc_test):.10f}"
        )
    print(msg)
    if log is not None:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print(msg, file=log)

    # Fairness summary for test
    bin_stats = _compute_bin_stats(bin_labels, bin_outputs)
    fairness = _compute_fairness_from_bins(bin_stats)
    _print_fairness_summary(bin_stats, fairness, prefix="Test", log_handle=log)

    del loss_test
    del auc_test
    del acc_test
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()


best_val_loss = float('inf')
best_epoch = -1

print(f'Starting training for {args.epochs} epochs...')
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch

print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_dir:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

test()
if log is not None:
    print(save_dir)
    log.close()