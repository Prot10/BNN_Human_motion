import time
import torch
import numpy as np
from tqdm import tqdm
from funcs.utils.data_utils import *
from ..loss import mpjpe_error



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def test(model, ckpt_path, test_loader, input_n,
         output_n, actions_to_consider_test='all'):

    model.load_state_dict(torch.load(ckpt_path))
    print('Model loaded')

    model = model.to(device)
    model.eval()
    accum_loss = 0
    n_batches = 0
    actions = define_actions(actions_to_consider_test)
    dim_used = np.array([ 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                          46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                          75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92 ])
    totalll = 0
    counter = 0

    for action in actions:
      running_loss = 0
      n = 0

      with torch.no_grad():
          for cnt, batch in enumerate(test_loader):
              batch = batch.float().to(device)
              batch_dim = batch.shape[0]
              n += batch_dim

              sequences_train = torch.cat((torch.zeros(*batch[:, :1, dim_used].size()).to(device), batch[:, 1:input_n, dim_used] - batch[:, :input_n-1, dim_used]), 1)
              sequences_gt = batch[:, input_n:input_n + output_n, dim_used]

              running_time = time.time()
              sequences_predict, kl_loss = model(sequences_train)
              sequences_predict[:, 1:output_n, :] = sequences_predict[:, 1:output_n, :] + sequences_predict[:, :(output_n-1), :]
              sequences_predict = (sequences_predict + batch[:, (input_n-1):input_n, dim_used])
              loss1 = mpjpe_error(sequences_predict, sequences_gt)
              loss = loss1 + kl_loss / batch_dim

              totalll += time.time()-running_time
              counter += 1

              running_loss += loss*batch_dim
              accum_loss += loss*batch_dim

          print(str(action),': ', str(np.round((running_loss/n).item(),1)))
          n_batches += n

    print('Average: ' + str(np.round((accum_loss/n_batches).item(),1)))
    print('Prediction time: ', totalll/counter)
    
    
    
def build_ci(x, alpha=0.1, bonferroni=True):
        # correct for the number of joints
        if bonferroni:
            alpha = alpha/x.shape[-2]

        # sample is dimension num_predictions x Batch x OutputFrames x Joints x 3
        mu = x.mean(dim=0, keepdim=True)
        diff = (x - mu)
        dev = torch.linalg.vector_norm(diff, dim=-1)
        dev = dev.numpy(force=True)

        return mu.squeeze(), np.quantile(dev, 1-alpha, axis=0)
    
    

def get_ci(model, ckpt_path, test_loader,
           input_n, output_n):
    
    model.load_state_dict(torch.load(ckpt_path))
    print('Model loaded')
    
    ci_batch = []
    mu_batch = []
    isin_batch = []
    
    dim_used = np.array([ 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                          46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                          75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92 ])

    with torch.no_grad():

        n = 0
        for _, batch in tqdm(enumerate(test_loader)):

            batch = batch.float().to(device)
            batch_dim = batch.shape[0]
            n += batch_dim

            sequences_train = torch.cat((torch.zeros(*batch[:, :1, dim_used].size()).to(device), batch[:, 1:input_n, dim_used] - batch[:, :input_n-1, dim_used]), 1)

            sequences_gt = batch[:, input_n:input_n + output_n, dim_used]
            sequences_gt = sequences_gt.view(sequences_gt.shape[0], sequences_gt.shape[1], 22, 3)

            sequences, _, _ = model(sequences_train)

            m, c = build_ci(sequences.permute(0, 2, 1, 3, 4))
            mu_batch.append(m)
            ci_batch.append(c)

            dev = torch.linalg.vector_norm(m - sequences_gt, dim=-1).numpy(force=True)
            isin = dev < c
            isin_batch.append(isin)

    mu = torch.concatenate(mu_batch, axis=0)
    ci = np.concatenate(ci_batch, axis=0)
    isin = np.concatenate(isin_batch, axis=0)
    
    return mu, ci, isin