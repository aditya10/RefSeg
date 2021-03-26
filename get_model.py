import CMPC_model
import CMPC_model_loss
# import CMPC_model_graphmod
# import CMPC_model_graphmod_2
# import CMPC_model_graphmod_while
# import CMPC_model_graphmod_duplicated
import CMPC_model_graphmod_dup_loss
import CMPC_model_refine_1
i#mport CMPC_model_refine_2
import CMPC_model_graphmod_3
import CMPC_model_extras

def get_segmentation_model(name, **kwargs):
    model = eval(name).LSTM_model(**kwargs)
    return model
