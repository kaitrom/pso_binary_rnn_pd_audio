"""
Utilitários para carregamento e preparação dos dados de Parkinson
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import config

def load_parkinson_data():
    """
    Carrega o dataset de Parkinson
    
    Returns:
        pd.DataFrame: Dataset carregado
    """
    try:
        df = pd.read_csv(config.DATA_CONFIG['dataset_path'])
        
        # Separar registros - retirando dados synth
        df_phon = df[df['name'].str.startswith('phon')]
        df_synth = df[df['name'].str.startswith('synth')]
        df = df_phon.copy()
        
        # Remove a coluna 'name' se existir
        if 'name' in df.columns:
            df = df.drop('name', axis=1)
        
        # Converter status para binário se necessário
        if 'status' in df.columns:
            # Se status tem valores contínuos, converter para binário
            if df['status'].nunique() > 2:
                # Usar threshold de 0.5 para converter para binário
                df['status'] = (df['status'] > 0.5).astype(int)
                print(f"⚠️ Status convertido para binário usando threshold 0.5")
                print(f"   Distribuição: {df['status'].value_counts().to_dict()}")
            
        return df
    except Exception as e:
        raise Exception(f"Erro ao carregar dataset: {str(e)}")

def prepare_data(df, test_size=None, random_state=None):
    """
    Prepara os dados para treinamento
    
    Args:
        df (pd.DataFrame): Dataset original
        test_size (float): Proporção do conjunto de teste
        random_state (int): Seed para reprodutibilidade
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    if test_size is None:
        test_size = config.DATA_CONFIG['test_size']
    if random_state is None:
        random_state = config.DATA_CONFIG['random_state']
    
    # Separar features e target
    target_col = config.DATA_CONFIG['target_column']
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # Normalizar os dados
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler

def prepare_data_for_cv(df, random_state=None):
    """
    Prepara os dados para validação cruzada
    
    Args:
        df (pd.DataFrame): Dataset original
        random_state (int): Seed para reprodutibilidade
        
    Returns:
        tuple: (X, y, scaler)
    """
    if random_state is None:
        random_state = config.DATA_CONFIG['random_state']
    
    # Separar features e target
    target_col = config.DATA_CONFIG['target_column']
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Embaralhar os dados
    X, y = shuffle(X, y, random_state=random_state)
    
    # Normalizar os dados
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values, scaler

def get_stratified_kfold(n_splits=5, random_state=None):
    """
    Retorna um objeto StratifiedKFold
    
    Args:
        n_splits (int): Número de folds
        random_state (int): Seed para reprodutibilidade
        
    Returns:
        StratifiedKFold: Objeto para validação cruzada estratificada
    """
    if random_state is None:
        random_state = config.EVAL_CONFIG['random_state']
        
    return StratifiedKFold(
        n_splits=n_splits, 
        shuffle=True, 
        random_state=random_state
    )

def get_data_info(df):
    """
    Retorna informações sobre o dataset
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        dict: Informações do dataset
    """
    target_col = config.DATA_CONFIG['target_column']
    
    info = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,  # excluindo target
        'target_distribution': df[target_col].value_counts().to_dict(),
        'feature_names': [col for col in df.columns if col != target_col],
        'missing_values': df.isnull().sum().sum()
    }
    
    return info

def validate_data(df):
    """
    Valida se o dataset está no formato esperado
    
    Args:
        df (pd.DataFrame): Dataset para validar
        
    Returns:
        bool: True se válido, False caso contrário
    """
    target_col = config.DATA_CONFIG['target_column']
    
    # Verificar se a coluna target existe
    if target_col not in df.columns:
        raise ValueError(f"Coluna target '{target_col}' não encontrada no dataset")
    
    # Verificar se há valores nulos
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contém valores nulos")
    
    # Verificar se target é binário
    unique_targets = df[target_col].unique()
    if len(unique_targets) != 2:
        raise ValueError(f"Target deve ser binário. Valores únicos encontrados: {unique_targets}")
    
    # Verificar se há pelo menos algumas amostras de cada classe
    target_counts = df[target_col].value_counts()
    if target_counts.min() < 5:
        raise ValueError("Muito poucas amostras de uma das classes")
    
    return True

