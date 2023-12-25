import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class testing():
    
    
    def __init__(self, model, ckpt_path):
    

        
        

    def test(model, ckpt_path):

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
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal*3, joint_equal*3+1, joint_equal*3+2))
        totalll = 0
        counter = 0

        for action in actions:
        running_loss = 0
        n = 0
        dataset_test = datasets.Datasets(path, input_n, output_n, skip_rate, split=2, actions=[action])
        test_loader = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0, pin_memory=True)

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
        
        



    num_predictions = 200
    model = Model(num_channels=num_channels,
                num_frames_out=output_n,
                old_frames=input_n,
                num_joints=num_joints,
                num_heads=num_heads,
                num_predictions=num_predictions,
                drop=dropout)
    path = './data/h3.6m/h3.6m/dataset'
    skip_rate = 1
    batch_size_test = 8
    actions_to_consider_test = 'all'
    ckpt_path = './checkpoints/Best_checkpoint.pt'

    test(model, ckpt_path)