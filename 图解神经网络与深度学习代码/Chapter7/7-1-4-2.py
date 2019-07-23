import numpy as np
from scipy import signal
I=np.array([3,4,1,5,6],np.float32)
K=np.array([-1,-2,2,1],np.float32)
#"¾í»ýºËK·­×ª180¶È"
K_reverse=np.flip(K,0)
#r=np.convolve(I,K_reverse,mode='full')
r=signal.convolve(I,K_reverse,mode='full')
print(r)