from model import BotDemineur
from utils import Transition, ReplayMemory
from env import DemineurInterface

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10



