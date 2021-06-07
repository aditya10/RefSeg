from models import *

model = eval('CMPC_model').LSTM_model()

def get_segmentation_model(name, **kwargs):
    model = eval(name).LSTM_model(**kwargs)
    return model
