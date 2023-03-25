from .td3 import TD3
from .sac import SAC
from .sacd import SACD
from .eaacd import EAACD
from .dagger import DAgger
from .ADVISORd import ADVISORd
from .ADVISORc import ADVISORc
from .ELFd import ELFd
from .ELFc import ELFc
from .daggerc import DAggerc
from .eaac import EAAC
from .A2D import A2D

RL_ALGORITHMS = {
    TD3.name: TD3,
    SAC.name: SAC,
    SACD.name: SACD,
    EAACD.name: EAACD,
    DAgger.name: DAgger,
    ADVISORd.name: ADVISORd,
    ADVISORc.name: ADVISORc,
    ELFd.name: ELFd,
    ELFc.name: ELFc,
    DAggerc.name: DAggerc,
    EAAC.name: EAAC,
    A2D.name: A2D
}
