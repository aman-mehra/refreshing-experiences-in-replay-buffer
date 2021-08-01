import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SimpleICMModel(nn.Module):
    def __init__(self, action_space):
        super(ICMModel, self).__init__()

        feature_output = 32*3*3
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            Flatten(),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(feature_output * 2, 256),
            nn.ELU(),
            nn.Linear(256, action_space)
        )

        self.forward_net = nn.Sequential(
            nn.Linear(action_space + feature_output, feature_output),
            nn.ELU(),
            nn.Linear(feature_output, feature_output),
            nn.ELU(),
            nn.Linear(feature_output, feature_output),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs, predict=False):

        state, next_state, action = inputs

        encoded_state = self.feature(state)
        encoded_next_state = self.feature(next_state)

        # predict next state
        pred_next_state = torch.cat((encoded_state, action), 1)
        pred_next_state = self.forward_net(pred_next_state)

        # For intermediate states where inverse model is not be trained
        if predict:
            return encoded_next_state, pred_next_state, None

        # predict action
        pred_action = torch.cat((encoded_state, encoded_next_state), 1)
        pred_action = self.inverse_net(pred_action)


        return encoded_next_state, pred_next_state, pred_action


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, net_type):
        super(FeatureExtractor, self).__init__()

        if net_type == "conv":
            feature_output = 32*3*3
            self.network = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                Flatten()
            )
        elif net_type == "fc":
            self.network = nn.Sequential(
                nn.Linear(in_channels,32),
                nn.ReLU(),
                nn.Linear(32,16),
            )

        # for p in self.modules():
        #     if isinstance(p, nn.Conv2d):
        #         init.orthogonal_(p.weight)
        #         # init.kaiming_uniform_(p.weight)
        #         p.bias.data.zero_()

    def forward(self, inputs):
        features = self.network(inputs)
        return features



class A2C(nn.Module):
    def __init__(self, action_space, net_type):
        super(A2C, self).__init__()

        if net_type == "conv":
            feature_output = 32*3*3
            self.policy = nn.Sequential(
                nn.Linear(feature_output,256),
                nn.ELU(),
                nn.Linear(256,128),
                nn.ELU(),
                nn.Linear(128,action_space),
            )

            self.value = nn.Sequential(
                nn.Linear(feature_output,256),
                nn.ELU(),
                nn.Linear(256,128),
                nn.ELU(),
                nn.Linear(128,1)
            )
        elif net_type == "fc":
            feature_output = 16
            self.policy = nn.Sequential(
                nn.Linear(feature_output,16),
                nn.ReLU(),
                nn.Linear(16,action_space),
            )

            self.value = nn.Sequential(
                nn.Linear(feature_output,16),
                nn.ReLU(),
                nn.Linear(16,1),
            )

        # for p in self.modules():
        #     if isinstance(p, nn.Linear):
        #         init.orthogonal_(p.weight)
        #         # init.kaiming_uniform_(p.weight, a=1.0)
        #         p.bias.data.zero_()

    def forward(self, cur_state, next_state, predict=False):
        action_probs = self.policy(cur_state)
        if predict:
            return action_probs, None, None
        cur_state_value = self.value(cur_state)
        next_state_value = self.value(next_state)
        return action_probs, cur_state_value, next_state_value


class ICMModule(nn.Module):
    def __init__(self, action_space, net_type):
        super(ICMModule, self).__init__()


        if net_type == "conv":

            feature_output = 32*3*3

            self.inverse_net = nn.Sequential(
                nn.Linear(feature_output * 2, 256),
                nn.ELU(),
                nn.Linear(256, action_space),
                # nn.Softmax()
            )

            self.forward_net = nn.Sequential(
                nn.Linear(action_space + feature_output, feature_output),
                nn.ELU(),
                nn.Linear(feature_output, feature_output),
                nn.ELU(),
                nn.Linear(feature_output, feature_output),
            )

        elif net_type == "fc":

            feature_output = 16

            self.inverse_net = nn.Sequential(
                nn.Linear(feature_output * 2, 32),
                nn.ReLU(),
                nn.Linear(32, action_space),
                # nn.Softmax()
            )

            self.forward_net = nn.Sequential(
                nn.Linear(action_space + feature_output, feature_output),
                nn.ReLU(),
                nn.Linear(feature_output, feature_output),
            )

        # for p in self.modules():
        #     if isinstance(p, nn.Conv2d):
        #         init.orthogonal_(p.weight)
        #         # init.kaiming_uniform_(p.weight)
        #         p.bias.data.zero_()

        #     if isinstance(p, nn.Linear):
        #         init.orthogonal_(p.weight)
        #         # init.kaiming_uniform_(p.weight, a=1.0)
        #         p.bias.data.zero_()

    def forward(self, inputs, predict=False):

        state, next_state, action = inputs

        # predict next state
        pred_next_state = torch.cat((state, action), 1)
        pred_next_state = self.forward_net(pred_next_state)

        # For intermediate states where inverse model is not be trained
        if predict:
            return pred_next_state, None

        # predict action
        pred_action = torch.cat((state, next_state), 1)
        pred_action = self.inverse_net(pred_action)


        return pred_next_state, pred_action


class DQN(nn.Module):

    def __init__(self, action_space, input_dim=84, in_channels=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        conv_dim = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_dim,8,4),4,2),3,1)
        linear_input_size = conv_dim * conv_dim * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, action_space)

        # for p in self.modules():
        #     if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
        #         init.orthogonal_(p.weight)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)



class DQN_FC(nn.Module):

    def __init__(self, action_space, input_dim=16):
        super(DQN_FC, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        # self.bn1 = nn.BatchNorm1d(32)
        self.head = nn.Linear(64, action_space)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.tanh(self.layer1(x))
        return self.head(x)