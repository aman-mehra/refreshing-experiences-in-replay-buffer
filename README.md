# refreshing-experiences-in-replay-buffer
A comparison and analysis of a refreshing strategy for experience replays.

## Background
Recent advancements in replay buffers have sought to make reinforcement learning more sample efficient by proposing schemes that sample transitions which are more *on-policy* with respect to the current agent policy. Consquently, complex methods of using KL divergene based sampling heuristics and learnable sampling policies have been proposed. 

We instead ask the question - ***can refreshing the contents of a sampled transition to make it align with the current policy achieve improved sample efficient and thereby faster convergence?*** The answer turns out be ***yes***.

## Proposed Approach
The proposed approach tackles the problem of refreshing as follows - given a transition (*S<sub>t</sub>*, *A<sub>t</sub>*, *R<sub>t</sub>*, *S<sub>t+1</sub>*) and the current agent policy _&pi;_, before a training epoch we replace *A<sub>t</sub>* with *A<sup>new</sup><sub>t</sub> = argmax<sub>a</sub> &pi;(a | S<sub>t</sub>)*, which is the greedy action the agent would take were it in state *S<sub>t</sub>* at the current time step. Having updated *A<sub>t</sub>*. we now reguire updating *R<sub>t</sub>* and *S<sub>t+1</sub>*. In a tabular setting this is straight forward since passing *S<sub>t</sub>* and *A<sup>new</sup><sub>t</sub>* to the environment will provide us the new reward *R<sup>new</sup><sub>t</sub>* and next state *S<sup>new</sup><sub>t+1</sub>*. In a continuous state space, this is trickier and will require the use of a forward dynamic model to estimate the next state as well as a model to capture the new reward. We have described an algorithm to do this using intrinsic motivation [here](https://github.com/aman-mehra/refreshing-experiences-in-replay-buffer/blob/main/Reports/Deep%20Refresh%20Algorithm.pdf), but so far have results on only tabular state spaces.


