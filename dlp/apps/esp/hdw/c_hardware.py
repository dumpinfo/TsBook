import time
import ctypes
import numpy as np
from apps.esp.hdw.hdw_common import PUL as PUL
from apps.esp.hdw.hdw_common import Input_I as Input_I
from apps.esp.hdw.hdw_common import Input as Input
from apps.esp.hdw.hdw_common import HardwareInput as HardwareInput

class CHardwareInput(ctypes.Structure):
    _fields_ = HardwareInput._fields_
