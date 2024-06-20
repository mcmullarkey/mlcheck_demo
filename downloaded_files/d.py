#!/bin/sh
# Make this file a valid bash script,
# in order to make use of the local python environment.
# Source: https://unix.stackexchange.com/a/20895
if "true" : '''\'
then
exec python3 "$0" "$@"
exit 127
fi
'''

from sys import path as pythonpath
import pathlib
from subprocess import check_output
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

this_file_path = str(pathlib.Path(__file__).resolve().parent)
repo_root_path = str(
    pathlib.Path(
        check_output(
            '''
                cd {this_file_path}
                echo -n `git rev-parse --show-toplevel`
            '''.format(this_file_path=this_file_path),
            shell=True
        ).decode('utf-8')
    ).resolve()
)
pythonpath.append(repo_root_path)

from src.models.preprocessing.grf import GRF
from src.models.preprocessing.multiplier import Multiplier
from src.models.training.diehl_network import DiehlNetwork
from src.models.decoding.diehl_decoder import DiehlDecoder

from tensorflow.keras.datasets import mnist


pipeline = make_pipeline(
    MinMaxScaler(),
    GRF(
        n_fields=11,
    ),
    Multiplier(input_multiplication=8),
    DiehlNetwork(
        network_parameters={
            'epochs': 2,
            'high_rate': 56.34299850324169,
            'input_to_inh_connection_prob': 0.1,
            'intervector_pause': 150.0,
            'low_rate': 0.1,
            'number_of_exc_neurons': 400,
            'number_of_inh_neurons': 400,
            'one_vector_longtitude': 350.0,
            'weight_normalization_during_training': None
        },
        neuron_parameters={
            'exc_neurons': {
                'C_m': 100.0,
                'E_L': -65.0,
                'E_ex': 0.0,
                'E_in': -100.0,
                'I_e': 0.0,
                'Theta_plus': 0.05,
                'Theta_rest': -72.0,
                'V_m': -105.0,
                'V_reset': -65.0,
                'V_th': -52.0,
                't_ref': 5.0,
                'tau_m': 100.0,
                'tau_synE': 1.0,
                'tau_synI': 2.0,
                'tc_theta': 10000000.0
            },
            'inh_neurons': {
                'C_m': 10.0,
                'E_L': -60.0,
                'E_ex': 0.0,
                'E_in': -100.0,
                'I_e': 0.0,
                'Theta_plus': 0.0,
                'Theta_rest': -40.0,
                'V_m': -100.0,
                'V_reset': -45.0,
                'V_th': -40.0,
                't_ref': 2.0,
                'tau_m': 10.0,
                'tau_synE': 1.0,
                'tau_synI': 2.0,
                'tc_theta': 1e+20
            }
        },
        synapse_parameters={
            'exc_to_inh': {
                'synapse_model': 'static_synapse',
                'weight': 10.4
            },
            'inh_to_exc': {
                'synapse_model': 'static_synapse',
                'weight': -9.49551728204824
            },
            'input_to_exc': {
                'Wmax': 1.0,
                'delay': {
                   'parametertype': 'uniform',
                   'specs': {
                       'max': 10.0,
                       'min': 0.1
                    }
                },
                'synapse_model': 'stdp_synapse',
                'weight': {
                   'parametertype': 'uniform',
                   'specs': {
                        'max': 1.0,
                        'min': 0.0
                    }
                }
            },
            'input_to_inh': {
                'delay': {
                    'parametertype': 'uniform',
                    'specs': {
                        'max': 5.0,
                        'min': 0.1
                    }
                },
                'synapse_model': 'static_synapse',
                'weight': {
                    'parametertype': 'uniform',
                    'specs': {
                        'max': 0.2,
                        'min': 0.0
                    }
                }
            }
        }
    ),
    DiehlDecoder(),
)


(x_train,y_train),(x_test,y_test) = mnist.load_data()
X, y = x_train, y_train

# X = [
#     [0, 1],
#     [1, 0],
# ]
# y = [
#     0,
#     'a',
# ]

pipeline.fit(X, y)
pipeline.predict(X)
