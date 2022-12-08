from .policy_mlp import ModelFreeOffPolicy_MLP as Off_Policy_MLP
from .policy_rnn_mlp import ModelFreeOffPolicy_RNN_MLP as Policy_RNN_MLP
from .policy_rnn import ModelFreeOffPolicy_Separate_RNN as Off_Policy_Separate_RNN
from .policy_rnn_shared import ModelFreeOffPolicy_Shared_RNN as Policy_Shared_RNN

AGENT_CLASSES = {
    "Off_Policy_MLP": Off_Policy_MLP,
    # "Policy_RNN_MLP": Policy_RNN_MLP,
    "Off_Policy_Separate_RNN": Off_Policy_Separate_RNN,
    # "Policy_Shared_RNN": Policy_Shared_RNN,
}

assert Off_Policy_Separate_RNN.ARCH == Policy_Shared_RNN.ARCH

from enum import Enum


class AGENT_ARCHS(str, Enum):
    # inherit from str to allow comparison with str
    Markov = Off_Policy_MLP.ARCH
    Memory_Markov = Policy_RNN_MLP.ARCH
    Memory = Off_Policy_Separate_RNN.ARCH
