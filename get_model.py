import CMPC_model
import CMPC_model_loss
import CMPC_model_graphmod
import CMPC_model_graphmod_2
import CMPC_model_graphmod_while
import CMPC_model_graphmod_duplicated

def get_segmentation_model(name, **kwargs):
    model = eval(name).LSTM_model(**kwargs)
    return model
