import signal
import sys
# import ray

def signal_handler(sig, frame):
    print('Hi, You pressed Ctrl+C!')
    # ray.shutdown() # quit ray progresses
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
