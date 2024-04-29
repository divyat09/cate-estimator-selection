from data.lbidd import lbidd_main_loader
from data.acic_2016 import acic_2016_main_loader
from data.realcause import real_cause_main_loader
from data.orthogonal_ml_dgp import get_data_generator                

def load_dataset_obj(dataset: str, root_dir: str, seed: int=0):
    """
    Returns a dictionary containing the generative model for the dataset.

    Inputs:
        dataset_name: Name of the particular dataset
        root_dir: Root directory for the dataset
        seed: Random seed
    """

    res={}
    if dataset in ['twins', 'lalonde_psid1', 'lalonde_cps1']:
        gen_model = real_cause_main_loader(dataset_name=dataset, root_dir=root_dir, seed= seed)
        res['gen_model']= gen_model

    elif 'acic_2016' in dataset:
        gen_model = acic_2016_main_loader(dataset_idx=int(dataset.split('_')[-1]), root_dir=root_dir, seed= seed)
        res['gen_model']= gen_model

    elif 'lbidd' in dataset:
        gen_model = lbidd_main_loader(n='5k', dataroot= root_dir, dataset_idx=int(dataset.split('_')[1]))
        res['gen_model']= gen_model

    elif dataset in ['orthogonal_ml_dgp']:
        gen_model, _, tau_fn, _ = get_data_generator()
        res['gen_model']= gen_model
        res['tau_fn']= tau_fn

    return res


def sample_dataset(dataset_obj: dict,  case: str='train'):
    """
    Samples the dataset for training or evaluation from the input dataset object.

    Inputs:
        dataset_obj: Dictionary containing the generative model for the dataset
        case: Split of the dataset; 'train' or 'eval'    

    Returns:
        Dictionary containing the samples for the dataset along with the ATE and ITE in case of evaluation split
        w: Expected Shape (num_samples, covariate dimensions)
        t: Expected Shape (num_samples, 1)
        y: Expected Shape (num_samples, 1)
        ate: Float
        ite: Expected Shape (num_samples)        
    """

    gen_model= dataset_obj['gen_model']

    if case == 'train':
        train_w, train_t, train_y= gen_model['train']['w'], gen_model['train']['t'], gen_model['train']['y']
        return  {'w': train_w, 't': train_t, 'y': train_y}

    elif case == 'eval':
        eval_w, eval_t, eval_y, ate, ite= gen_model['test']['w'], gen_model['test']['t'], gen_model['test']['y'], \
                                            gen_model['test']['ate'], gen_model['test']['ite']

        return {'w': eval_w, 't': eval_t, 'y': eval_y, 'ate': ate, 'ite': ite}
