from .model import Model
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ..loss import mpjpe_error



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Training():
    
    def __init__(self, model, data_loader, vald_loader, input_n, output_n,
                 clip_grad=None, device=device, n_epochs=25, log_step=100, 
                 lr=1e-04, use_scheduler=True, milestones=[4, 8, 12, 16],
                 gamma=0.7, weight_decay=3e-04, use_wandb=False, save_and_plot=True):
    
        self.model = model
        self.data_loader = data_loader
        self.vald_loader = vald_loader
        self.input_n = input_n
        self.output_n = output_n
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.clip_grad = clip_grad
        self.device = device
        self.n_epochs = n_epochs
        self.log_step = log_step
        self.use_wandb = use_wandb
        self.save_and_plot = save_and_plot
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        
        
    def train(self):
            
        train_loss = []
        val_loss = []
        val_loss_best = 1000
        train_losses = []
        val_losses = []
        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                            46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                            75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

        for epoch in range(self.n_epochs):
            running_loss = 0
            n = 0

            self.model.train()
            for cnt, batch in tqdm(enumerate(self.data_loader)):
                batch = batch.float().to(self.device)
                batch_dim = batch.shape[0]
                n += batch_dim

                sequences_train = torch.cat((torch.zeros(*batch[:, :1, dim_used].size()).to(device), batch[:, 1:10, dim_used]-batch[:, :9, dim_used]), 1)
                sequences_gt = batch[:, self.input_n:self.input_n + self.output_n, dim_used]

                self.optimizer.zero_grad()

                sequences_predict, kl_loss = self.model(sequences_train)
                sequences_predict[:, 1:self.output_n, :] = sequences_predict[:, 1:self.output_n, :] + sequences_predict[:, :self.output_n-1, :]
                sequences_predict = (sequences_predict + batch[:, (self.input_n-1):self.input_n, dim_used])

                loss1 = mpjpe_error(sequences_predict, sequences_gt)
                kl_loss /= batch_dim
                loss = loss1 + kl_loss

                if self.use_wandb:
                    self.wandb.log({"train_mpjpe_error": loss1, "train_kl_loss": kl_loss, "train_total_loss": loss})

                if cnt % self.log_step == 0:
                    print('[Epoch: {:<2d} | Iteration: {:>5d} | Train ] MPJPE loss: {:.3f} | KL loss: {:.3f} | Total loss: {:.3f}'.format(epoch + 1, cnt + 1, loss1, kl_loss, loss.item()))

                loss.backward()

                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

                self.optimizer.step()
                running_loss += loss * batch_dim

            train_loss.append(running_loss.detach().cpu()/n)

            self.model.eval()
            with torch.no_grad():
                running_loss = 0
                n = 0
                for cnt, batch in tqdm(enumerate(self.vald_loader)):
                    batch = batch.float().to(device)
                    batch_dim = batch.shape[0]
                    n += batch_dim

                    sequences_train = torch.cat((torch.zeros(*batch[:, :1, dim_used].size()).to(device), batch[:, 1:self.input_n, dim_used] - batch[:, :self.input_n-1, dim_used]), 1)
                    sequences_gt = batch[:, self.input_n:self.input_n+self.output_n, dim_used]

                    sequences_predict, kl_loss = self.model(sequences_train)
                    sequences_predict[:, 1:self.output_n, :] = sequences_predict[:, 1:self.output_n, :] + sequences_predict[:, :(self.output_n-1), :]
                    sequences_predict = (sequences_predict + batch[:, (self.input_n-1):self.input_n, dim_used])
                    loss1 = mpjpe_error(sequences_predict, sequences_gt)
                    loss = loss1 + kl_loss / batch_dim

                    if self.use_wandb:
                        self.wandb.log({"val_mpjpe_error": loss1, "val_kl_loss": kl_loss, "val_total_loss": loss})

                    if cnt % self.log_step == 0:
                        print('\033[1m[Epoch: {:<2d} | Iteration: {:>5d} | Val   ] MPJPE loss: {:.3f} | KL loss: {:.3f} | Total loss: {:.3f}\033[0m'.format(epoch + 1, cnt + 1, loss1, kl_loss / batch_dim, loss.item()))

                    running_loss += loss * batch_dim

                val_loss.append(running_loss.detach().cpu()/n)

                if running_loss/n < val_loss_best:
                    val_loss_best = running_loss/n
                    torch.save(self.model.state_dict(), './checkpoints/Best_checkpoint.pt')
                    if self.use_wandb:
                        self.wandb.run.log_artifact('./checkpoints/Best_checkpoint.pt', name="Best_checkpoint")

                train_losses.append(train_loss[-1])
                val_losses.append(val_loss[-1])

        if self.use_scheduler:
            self.scheduler.step()

        epochs = list(np.arange(1, self.n_epochs+1))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Train Loss', marker='o')
        plt.plot(epochs, val_losses, label='Validation Loss', marker='o')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')

        plt.legend()

        plt.grid(True)
        plt.show()

        return train_losses, val_losses