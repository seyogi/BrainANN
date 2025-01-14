import torch
import torch.nn as nn
import pickle


class KalmanCTRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_net,
        mlp_net,
        Kalman_flag,
        noise_flag,
        dt=None,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.mlp_net = mlp_net
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.input2h = rnn_net.rnn.input2h
        self.h2h = rnn_net.rnn.h2h

        # Kalman
        self._A = rnn_net.rnn.h2h.weight
        self._C = rnn_net.fc.weight
        self._R = torch.eye(3) * 1
        self._Q = torch.eye(64) * 0
        self._cov = torch.eye(64) * 1
        self.Kalman_flag = Kalman_flag
        self.noise_flag = noise_flag

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden, tmp_activitys):
        # h_obs = nn.functional.tanh(self.input2h(input) + self.h2h(hidden))
        h_obs = torch.relu(self.input2h(input) + self.h2h(hidden))
        if self.noise_flag:
            h_obs = torch.mul(h_obs, torch.normal(mean=1, std=0.1, size=(1,64)))
        h_obs = hidden * (1 - self.alpha) + h_obs * self.alpha

        if len(tmp_activitys) == 5:
            ## Estimate by MLP
            _tmp = torch.cat(
                [
                    tmp_activitys[0],
                    tmp_activitys[1],
                    tmp_activitys[2],
                    tmp_activitys[3],
                    tmp_activitys[4],
                ],
                dim=1,
            )
            _tmp = torch.tensor(_tmp, dtype=float)
            h_mlp = self.mlp_net(_tmp).to(torch.float32)

            if self.Kalman_flag:
                ## Kalman Filter
                _precov = (
                    torch.matmul(torch.matmul(self._A, self._cov), self._A.T) + self._Q
                )
                _inv = torch.inverse(
                    torch.matmul(torch.matmul(self._C, _precov), self._C.T) + self._R
                )
                K = torch.matmul(torch.matmul(_precov, self._C.T), _inv)
                KC = torch.matmul(K, self._C)
                IKC = torch.eye(64) - KC
                self._cov = torch.matmul(IKC, _precov)
                #h_obs = (torch.matmul(torch.diag(torch.diag(IKC,0)), h_obs.T) + torch.matmul(torch.diag(torch.diag(KC,0)), h_mlp.T)).T
                #h_obs =  torch.relu((torch.matmul(IKC, h_obs.T) + torch.matmul(KC, h_mlp.T)).T)
                h_obs =  (torch.matmul(IKC, h_obs.T) + torch.matmul(KC, h_mlp.T)).T
        return h_obs

    def forward(self, input, hidden=None):
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        steps = range(input.size(0))
        tmp_activitys = []
        for i in steps:
            hidden = self.recurrence(input[i], hidden, tmp_activitys)

            tmp_activitys.append(hidden)
            if len(tmp_activitys) > 5:
                tmp_activitys.pop(0)

            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden


class KalmanRNNNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        rnn_net,
        mlp_net,
        Kalman_flag=True,
        noise_flag=False,
        **kwargs
    ):
        super().__init__()

        # Continuous time RNN
        self.rnn = KalmanCTRNN(
            input_size, hidden_size, rnn_net, mlp_net, Kalman_flag, noise_flag, **kwargs
        )

        # Add an output layer
        self.fc = rnn_net.fc

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output
