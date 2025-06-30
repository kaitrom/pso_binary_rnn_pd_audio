"""
Configurações para o sistema de otimização PSO para classificação de Parkinson
"""

# Configurações da Rede Neural
NN_CONFIG = {
    'min_layers': 1,
    'max_layers': 4,
    'min_neurons': 8,
    'max_neurons': 128,
    'min_lr': 1e-5,
    'max_lr': 1e-1,
    'epochs': 30,
    'batch_size': 32,
    'validation_split': 0.2,
    'early_stopping_patience': 5,
    'random_state': 42
}

# Configurações do PSO
PSO_CONFIG = {
    'n_particles': 20,
    'dimensions': 5,  # [n_layers, n1, n2, n3, learning_rate]
    'iters': 20,
    'experimentos': 32,
    'options': {
        'c1': 1.5,  # cognitive parameter
        'c2': 1.5,  # social parameter
        'w': 0.7    # inertia weight
    },
    'bounds': {
        'lower': [1, 8, 8, 8, 1e-5],      # [min_layers, min_neurons, min_neurons, min_neurons, min_lr]
        'upper': [4, 128, 128, 128, 1e-1]  # [max_layers, max_neurons, max_neurons, max_neurons, max_lr]
    },
    'random_state': 42
}

# Configurações do Banco de Dados
DB_NAME = 'pso_parkinson.db'

# Configurações do Dataset
DATA_CONFIG = {
    'dataset_path': 'dataset.csv',
    'target_column': 'status',
    'test_size': 0.2,
    'random_state': 42
}

# Configurações de Avaliação Final
EVAL_CONFIG = {
    'cv_folds': 5,
    'random_state': 42,
    'scoring_metrics': ['f1', 'accuracy', 'roc_auc']
}

# Configurações de Monitoramento
MONITORING_CONFIG = {
    'save_interval': 1,  # salvar a cada iteração
    'progress_bar': True,
    'verbose': True
}

