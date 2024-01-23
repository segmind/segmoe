from .main import SegMixPipeline
import sys

def create(args=sys.argv):
    pipe = SegMixPipeline(args[1])
    pipe.save_pretrained(args[2])