import argparse
from rnn_engine import Rnn_Engine


def main():
    print('RNN char model')
    rnn_engine = Rnn_Engine()
    #rnn_engine.build_model()
    rnn_engine.train()
    
    
if '__main__' == __name__:
    main()