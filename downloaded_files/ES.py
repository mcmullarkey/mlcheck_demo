import cma
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

data = pd.read_csv('./result.csv', low_memory=False)
target_columns = ['S_pX']
columns_drop = ['mof', 'nmolecule', 'nmolecule1', 'nmolecule2', 'nmolecule3', 'nmolecule4',
                'std_nmolecule', 'std_nmolecule1', 'std_nmolecule2', 'std_nmolecule3', 'std_nmolecule4',
                'atomic_mass', 'S_pX', 'S_mX', 'S_oX', 'S_EB', 'N_pX', 'N_mX', 'N_oX', 'N_EB']
X_original = data.drop(columns=columns_drop)
y_original = data[target_columns]

scale = MinMaxScaler()
X_scaled = scale.fit_transform(X_original)
X = pd.DataFrame(X_scaled, columns=X_original.columns).values

y = y_original.values
y = y.reshape((len(y),))

def closest_mof(x_ask, ids_acquired):
    distances_to_xs = np.linalg.norm(X - x_ask, axis=1)
    ids_sorted_by_closeness = np.argsort(distances_to_xs)
    for id_closest in ids_sorted_by_closeness:
        if id_closest not in ids_acquired:
            return id_closest

def evol_run(nb_iterations):
    x_init = X[np.random.choice(np.arange(len(X)), size=10, replace=False)[0]]

    get_index = []
    es = cma.CMAEvolutionStrategy(x0=x_init, sigma0=0.5)

    while len(get_index) < nb_iterations:

        xs_ask = es.ask()

        ids_ask = []
        for x_ask in xs_ask:
            index_ask = closest_mof(x_ask, get_index)
            ids_ask.append(index_ask)
            get_index.append(index_ask)


        es.tell(X[ids_ask], -y[ids_ask])
        print(f"y value: {np.max(y[get_index])}")

    return get_index

niter = 500
nruns = 100

es_res = {'get_index': []}

for r in range(nruns):
    print(f"\n\nRUN {r}")
    ids_acquired = evol_run(niter)
    es_res['get_index'].append(ids_acquired)

with open('es_results}.pkl', 'wb') as file:
    pickle.dump(es_res, file)
