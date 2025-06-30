"""
Utilitários para construção e treino da rede neural feedforward
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
import config
import warnings

# Suprimir warnings do TensorFlow
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

def decode_particle_position(position):
    """
    Decodifica a posição da partícula em parâmetros da rede neural
    
    Args:
        position (list): Posição da partícula [n_layers, n1, n2, n3, learning_rate]
        
    Returns:
        dict: Parâmetros decodificados da rede neural
    """
    # Extrair valores da posição
    n_layers_raw, n1_raw, n2_raw, n3_raw, lr_raw = position
    
    # Converter número de camadas para inteiro
    n_layers = max(1, min(4, int(round(n_layers_raw))))
    
    # Converter neurônios para inteiros
    neurons = []
    for neuron_raw in [n1_raw, n2_raw, n3_raw]:
        neurons.append(max(8, min(128, int(round(neuron_raw)))))
    
    # Usar apenas o número necessário de camadas
    neurons = neurons[:n_layers]
    
    # Garantir que learning rate está nos limites
    learning_rate = max(config.NN_CONFIG['min_lr'], 
                       min(config.NN_CONFIG['max_lr'], lr_raw))
    
    return {
        'n_layers': n_layers,
        'neurons': neurons,
        'learning_rate': learning_rate
    }

def create_neural_network(input_dim, params):
    """
    Cria uma rede neural feedforward baseada nos parâmetros
    
    Args:
        input_dim (int): Dimensão de entrada
        params (dict): Parâmetros da rede neural
        
    Returns:
        tf.keras.Model: Modelo da rede neural
    """
    model = models.Sequential()
    
    # Camada de entrada
    model.add(layers.Dense(
        params['neurons'][0], 
        activation='relu', 
        input_shape=(input_dim,),
        name='hidden_1'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    # Camadas ocultas adicionais
    for i in range(1, params['n_layers']):
        if i < len(params['neurons']):
            units = params['neurons'][i]
        else:
            units = params['neurons'][-1]  # Repete o último valor
        
        model.add(layers.Dense(
            units,
            activation='relu',
            name=f'hidden_{i+1}'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
    
    # Camada de saída para classificação binária
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    # Compilar o modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_neural_network(model, X_train, y_train, X_val=None, y_val=None, 
                        epochs=None, batch_size=None, verbose=0):
    """
    Treina a rede neural
    
    Args:
        model: Modelo da rede neural
        X_train: Dados de treino
        y_train: Labels de treino
        X_val: Dados de validação (opcional)
        y_val: Labels de validação (opcional)
        epochs: Número de épocas
        batch_size: Tamanho do batch
        verbose: Nível de verbosidade
        
    Returns:
        tf.keras.callbacks.History: Histórico do treinamento
    """
    if epochs is None:
        epochs = config.NN_CONFIG['epochs']
    if batch_size is None:
        batch_size = config.NN_CONFIG['batch_size']
    
    callbacks = []
    
    # Early stopping se dados de validação fornecidos
    if X_val is not None and y_val is not None:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config.NN_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stopping)
        
        validation_data = (X_val, y_val)
    else:
        validation_data = None
    
    # Treinar o modelo
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo e retorna métricas
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste
        y_test: Labels de teste
        
    Returns:
        dict: Métricas de avaliação
    """
    # Fazer predições
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calcular métricas
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'f1_score': f1,
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba.flatten()
    }

def fitness_function(position, X_train, y_train, X_val, y_val, input_dim):
    """
    Função de fitness para o PSO (minimizar 1 - F1-score)
    
    Args:
        position: Posição da partícula
        X_train: Dados de treino
        y_train: Labels de treino
        X_val: Dados de validação
        y_val: Labels de validação
        input_dim: Dimensão de entrada
        
    Returns:
        float: Valor de fitness (1 - F1-score)
    """
    try:
        # Decodificar parâmetros
        params = decode_particle_position(position)
        
        # Criar e treinar modelo
        model = create_neural_network(input_dim, params)
        train_neural_network(model, X_train, y_train, X_val, y_val, verbose=0)
        
        # Avaliar modelo
        metrics = evaluate_model(model, X_val, y_val)
        
        # Retornar fitness (minimizar 1 - F1)
        fitness = 1.0 - metrics['f1_score']
        
        # Limpar memória
        del model
        tf.keras.backend.clear_session()
        
        return fitness
        
    except Exception as e:
        # Em caso de erro, retornar fitness alto (pior)
        print(f"Erro na função de fitness: {str(e)}")
        print(traceback.print_exc())
        return 1.0

def train_final_model(position, X_train, y_train, input_dim, epochs=None):
    """
    Treina o modelo final com os melhores parâmetros
    
    Args:
        position: Melhor posição encontrada pelo PSO
        X_train: Dados de treino
        y_train: Labels de treino
        input_dim: Dimensão de entrada
        epochs: Número de épocas (opcional)
        
    Returns:
        tf.keras.Model: Modelo treinado
    """
    # Decodificar parâmetros
    params = decode_particle_position(position)
    
    # Criar modelo
    model = create_neural_network(input_dim, params)
    
    # Treinar modelo
    if epochs is None:
        epochs = config.NN_CONFIG['epochs']
        
    train_neural_network(model, X_train, y_train, epochs=epochs, verbose=1)
    
    return model, params

def get_model_summary(params):
    """
    Retorna um resumo dos parâmetros do modelo
    
    Args:
        params (dict): Parâmetros do modelo
        
    Returns:
        str: Resumo formatado
    """
    summary = f"""
    Arquitetura da Rede Neural:
    - Número de camadas ocultas: {params['n_layers']}
    - Neurônios por camada: {params['neurons']}
    - Taxa de aprendizado: {params['learning_rate']:.6f}
    """
    return summary

