import CMPC_model
import CMPC_model_loss
import CMPC_model_graphmod

def get_segmentation_model(name, **kwargs):
    model = eval(name).LSTM_model(**kwargs)
    return model
