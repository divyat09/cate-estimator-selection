import os
import pickle
import numpy as np

from data.loading import load_from_folder

#To facilitate tweaking the heterogenity knobs etc
def real_cause_main_loader(dataset_name: str= 'twins', root_dir: str= '/scratch/cate_eval_analysis/', seed: int=0):
    """
    Samples the dataset for training or evaluation.

    Inputs:
        dataset_name: Name of the particular dataset
        root_dir: Root directory for the dataset
        seed: Random seed

    Returns:
        Dictionary containing the samples for the dataset along with the ATE and ITE in case of evaluation split        
        w: Expected Shape (num_samples, covariate dimensions)
        t: Expected Shape (num_samples, 1)
        y: Expected Shape (num_samples, 1)
        ate: Float
        ite: Expected Shape (num_samples) 
    """

    base_dir= os.path.join(os.path.expanduser('~'), root_dir,  dataset_name, 'seed_' + str(seed), '')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    fname= base_dir + 'data.p'
    if os.path.exists(fname):
        print('Loading data from a saved file for Real Cause datasets')
        final_data= pickle.load(open(fname, "rb"))
        return final_data    

    # Train/Val split
    print('Generating data for Real Cause datasets')
    gen_model, _ = load_from_folder(dataset=dataset_name)
    final_data = {'train': {}, 'test': {}}    
    for case in ['train', 'test']:
        w, t, (y0, y1) = gen_model.sample(dataset=case, seed=seed, ret_counterfactuals=True)
        y = y1 * t + y0 * (1 - t)

        final_data[case]['w']= w
        final_data[case]['t']= t 
        final_data[case]['y']= y 

        #No need to generate ATE and ITE for training data
        if case == 'train':
            continue

        #Generate ATE and ITE for evaluation data
        ate = gen_model.ate(w=w, noisy=True)
        ite = gen_model.ite(w=w, noisy=True, seed=seed).squeeze()

        final_data[case]['ate']= ate
        final_data[case]['ite']= ite

    #Save data for easy future access
    pickle.dump(final_data, open(fname, "wb") )
    
    return final_data