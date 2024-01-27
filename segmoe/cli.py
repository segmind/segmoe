from .main import SegMoEPipeline
import sys

def create(args=sys.argv):
    pipe = SegMoEPipeline(args[1])
    pipe.save_pretrained(args[2])