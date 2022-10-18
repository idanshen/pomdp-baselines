from .policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP
from .policy_rnn_mlp import ModelFreeOffPolicy_RNN_MLP as Policy_RNN_MLP
from .policy_rnn import ModelFreeOffPolicy_Separate_RNN as Off_Policy_Separate_RNN
from .policy_rnn import ModelFreeOnPolicy_Separate_RNN as On_Policy_Separate_RNN
from .policy_rnn_shared import ModelFreeOffPolicy_Shared_RNN as Policy_Shared_RNN

AGENT_CLASSES = {
    "Policy_MLP": Policy_MLP,
    "Policy_RNN_MLP": Policy_RNN_MLP,
    "Off_Policy_Separate_RNN": Off_Policy_Separate_RNN,
    "On_Policy_Separate_RNN": On_Policy_Separate_RNN,
    "Policy_Shared_RNN": Policy_Shared_RNN,
}

assert Off_Policy_Separate_RNN.ARCH == Policy_Shared_RNN.ARCH

from enum import Enum


class AGENT_ARCHS(str, Enum):
    # inherit from str to allow comparison with str
    Markov = Policy_MLP.ARCH
    Memory_Markov = Policy_RNN_MLP.ARCH
    Memory = Off_Policy_Separate_RNN.ARCH

class TRAIN_TYPES(Enum):
    # inherit from str to allow comparison with str
    OFF_POLICY = 0
    ON_POLICY = 1
