import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers import LinearReparameterization as BayesianLinear
from bayesian_torch.layers import Conv1dReparameterization as BayesianConv1d



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Decoder(nn.Module):


    def __init__(self, hidden_dim, num_predictions, num_layers=1, conv_channels=128, kernel_size=3):
        
        super(Decoder, self).__init__()

        # Layers definition
        self.gru = nn.GRU(66, hidden_dim, num_layers=num_layers)
        self.conv1d = BayesianConv1d(hidden_dim, conv_channels, kernel_size, padding=1)
        self.out = BayesianLinear(conv_channels, 66)
        
        # Number of predictions to make
        self.num_predictions = num_predictions


    def forward(self, hidden, num_steps):
        
        # Initialize parameters and variables
        batch_size = hidden.size(1)
        kl_loss = 0
        input = torch.zeros((batch_size, 66), dtype=torch.float).unsqueeze(0).to(device)
        outputs = torch.zeros((num_steps, batch_size, 66), dtype=torch.float).to(device)

        # Loop over the number of steps
        for t in range(num_steps):
            
            decoder_output, _ = self.gru(input, hidden)
            
            decoder_output = decoder_output.permute(0, 2, 1)
            decoder_output, kl_1 = self.conv1d(decoder_output)
            decoder_output = F.relu(decoder_output)
            
            decoder_output = decoder_output.permute(0, 2, 1)
            decoder_output, kl_2 = self.out(decoder_output[-1])
            
            # Store the current output in the outputs tensor
            outputs[t] = decoder_output
            
            # Update the input tensor with the current output
            input = torch.cat((input, decoder_output.unsqueeze(0)), 0)
            
            # Total KL loss from both Bayesian layers
            kl_loss += kl_1 + kl_2

        return outputs, kl_loss