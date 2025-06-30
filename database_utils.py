"""
Utilitários para criação e manipulação do banco de dados SQLite
"""

import sqlite3
import pandas as pd
import numpy as np
import config
import os
from typing import List, Dict, Any

def create_database():
    """
    Cria o banco de dados e as tabelas necessárias
    
    Returns:
        str: Caminho do banco de dados criado
    """
    db_path = config.DB_NAME
    
    # Remover banco existente se houver
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Criar tabela pso_resultados
    cursor.execute('''
        CREATE TABLE pso_resultados (
            num_experimento INTEGER,
            num_iteracao INTEGER,
            num_particula INTEGER,
            pos_camada REAL,
            pos_n1 REAL,
            pos_n2 REAL,
            pos_n3 REAL,
            pos_lr REAL,
            vel_camada REAL,
            vel_n1 REAL,
            vel_n2 REAL,
            vel_n3 REAL,
            vel_lr REAL,
            pbest_camada REAL,
            pbest_n1 REAL,
            pbest_n2 REAL,
            pbest_n3 REAL,
            pbest_lr REAL,
            num_camadas INTEGER,
            f1_score REAL,
            peso REAL,
            int_best INTEGER
        )
    ''')
    
    # Criar tabela pso_execucao
    cursor.execute('''
        CREATE TABLE pso_execucao (
            num_experimento INTEGER,
            tempo_total_seg REAL,
            tempo_medio_iteracao REAL,
            tempo_medio_treino_particula REAL,
            uso_medio_cpu REAL,
            uso_max_memoria_mb REAL,
            uso_disco_mb REAL,
            total_iteracoes INTEGER
        )
    ''')
    
    # Criar índices para melhor performance
    cursor.execute('CREATE INDEX idx_experimento_iteracao ON pso_resultados(num_experimento, num_iteracao)')
    cursor.execute('CREATE INDEX idx_experimento_exec ON pso_execucao(num_experimento)')
    
    conn.commit()
    conn.close()
    
    print(f"Banco de dados criado: {db_path}")
    return db_path

def insert_pso_results_batch(experiment_num: int, iteration_num: int, 
                           particles_data: List[Dict[str, Any]]):
    """
    Insere dados de todas as partículas de uma iteração em lote
    
    Args:
        experiment_num (int): Número do experimento
        iteration_num (int): Número da iteração
        particles_data (list): Lista com dados de cada partícula
    """
    conn = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    
    # Preparar dados para inserção
    insert_data = []
    for particle_data in particles_data:
        row = (
            experiment_num,
            iteration_num,
            particle_data['particle_id'],
            particle_data['position'][0],  # pos_camada
            particle_data['position'][1],  # pos_n1
            particle_data['position'][2],  # pos_n2
            particle_data['position'][3],  # pos_n3
            particle_data['position'][4],  # pos_lr
            particle_data['velocity'][0],  # vel_camada
            particle_data['velocity'][1],  # vel_n1
            particle_data['velocity'][2],  # vel_n2
            particle_data['velocity'][3],  # vel_n3
            particle_data['velocity'][4],  # vel_lr
            particle_data['pbest_position'][0],  # pbest_camada
            particle_data['pbest_position'][1],  # pbest_n1
            particle_data['pbest_position'][2],  # pbest_n2
            particle_data['pbest_position'][3],  # pbest_n3
            particle_data['pbest_position'][4],  # pbest_lr
            particle_data['num_layers'],
            particle_data['f1_score'],
            particle_data['fitness'],  # peso (fitness)
            1 if particle_data['is_gbest'] else 0  # int_best
        )
        insert_data.append(row)
    
    # Inserir em lote
    cursor.executemany('''
        INSERT INTO pso_resultados (
            num_experimento, num_iteracao, num_particula,
            pos_camada, pos_n1, pos_n2, pos_n3, pos_lr,
            vel_camada, vel_n1, vel_n2, vel_n3, vel_lr,
            pbest_camada, pbest_n1, pbest_n2, pbest_n3, pbest_lr,
            num_camadas, f1_score, peso, int_best
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', insert_data)
    
    conn.commit()
    conn.close()

def insert_experiment_summary(experiment_num: int, summary_data: Dict[str, Any]):
    """
    Insere resumo de um experimento completo
    
    Args:
        experiment_num (int): Número do experimento
        summary_data (dict): Dados resumidos do experimento
    """
    conn = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO pso_execucao (
            num_experimento, tempo_total_seg, tempo_medio_iteracao,
            tempo_medio_treino_particula, uso_medio_cpu, uso_max_memoria_mb,
            uso_disco_mb, total_iteracoes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        experiment_num,
        summary_data['tempo_total_seg'],
        summary_data['tempo_medio_iteracao'],
        summary_data['tempo_medio_treino_particula'],
        summary_data['uso_medio_cpu'],
        summary_data['uso_max_memoria_mb'],
        summary_data['uso_disco_mb'],
        summary_data['total_iteracoes']
    ))
    
    conn.commit()
    conn.close()

def get_best_particle_overall():
    """
    Retorna a melhor partícula de todos os experimentos
    
    Returns:
        dict: Dados da melhor partícula
    """
    conn = sqlite3.connect(config.DB_NAME)
    
    query = '''
        SELECT * FROM pso_resultados 
        WHERE f1_score = (SELECT MAX(f1_score) FROM pso_resultados)
        LIMIT 1
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) > 0:
        best_particle = df.iloc[0]
        return {
            'experiment': int(best_particle['num_experimento']),
            'iteration': int(best_particle['num_iteracao']),
            'particle': int(best_particle['num_particula']),
            'position': [
                best_particle['pos_camada'],
                best_particle['pos_n1'],
                best_particle['pos_n2'],
                best_particle['pos_n3'],
                best_particle['pos_lr']
            ],
            'f1_score': best_particle['f1_score'],
            'num_layers': int(best_particle['num_camadas'])
        }
    else:
        return None

def get_experiment_statistics():
    """
    Retorna estatísticas dos experimentos
    
    Returns:
        dict: Estatísticas dos experimentos
    """
    conn = sqlite3.connect(config.DB_NAME)
    
    # Estatísticas dos resultados
    results_query = '''
        SELECT 
            COUNT(DISTINCT num_experimento) as total_experiments,
            COUNT(DISTINCT num_iteracao) as iterations_per_experiment,
            COUNT(*) as total_particles,
            AVG(f1_score) as avg_f1_score,
            MAX(f1_score) as best_f1_score,
            MIN(f1_score) as worst_f1_score,
            STDDEV(f1_score) as std_f1_score
        FROM pso_resultados
    '''
    
    # Estatísticas de execução
    exec_query = '''
        SELECT 
            AVG(tempo_total_seg) as avg_total_time,
            AVG(tempo_medio_iteracao) as avg_iteration_time,
            AVG(tempo_medio_treino_particula) as avg_particle_time,
            AVG(uso_medio_cpu) as avg_cpu_usage,
            AVG(uso_max_memoria_mb) as avg_max_memory,
            AVG(uso_disco_mb) as avg_disk_usage
        FROM pso_execucao
    '''
    
    results_df = pd.read_sql_query(results_query, conn)
    exec_df = pd.read_sql_query(exec_query, conn)
    
    conn.close()
    
    stats = {}
    if len(results_df) > 0:
        stats.update(results_df.iloc[0].to_dict())
    if len(exec_df) > 0:
        stats.update(exec_df.iloc[0].to_dict())
    
    return stats

def export_results_to_csv(output_dir: str = "."):
    """
    Exporta resultados para arquivos CSV
    
    Args:
        output_dir (str): Diretório de saída
    """
    conn = sqlite3.connect(config.DB_NAME)
    
    # Exportar tabela de resultados
    results_df = pd.read_sql_query("SELECT * FROM pso_resultados", conn)
    results_path = os.path.join(output_dir, "pso_resultados.csv")
    results_df.to_csv(results_path, index=False)
    
    # Exportar tabela de execução
    exec_df = pd.read_sql_query("SELECT * FROM pso_execucao", conn)
    exec_path = os.path.join(output_dir, "pso_execucao.csv")
    exec_df.to_csv(exec_path, index=False)
    
    conn.close()
    
    print(f"Resultados exportados para:")
    print(f"  - {results_path}")
    print(f"  - {exec_path}")

def get_convergence_data(experiment_num: int = None):
    """
    Retorna dados de convergência para análise
    
    Args:
        experiment_num (int): Número do experimento específico (opcional)
        
    Returns:
        pd.DataFrame: Dados de convergência
    """
    conn = sqlite3.connect(config.DB_NAME)
    
    if experiment_num is not None:
        query = '''
            SELECT num_iteracao, MAX(f1_score) as best_f1_score, AVG(f1_score) as avg_f1_score
            FROM pso_resultados 
            WHERE num_experimento = ?
            GROUP BY num_iteracao
            ORDER BY num_iteracao
        '''
        df = pd.read_sql_query(query, conn, params=[experiment_num])
    else:
        query = '''
            SELECT num_experimento, num_iteracao, MAX(f1_score) as best_f1_score, AVG(f1_score) as avg_f1_score
            FROM pso_resultados 
            GROUP BY num_experimento, num_iteracao
            ORDER BY num_experimento, num_iteracao
        '''
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df

def validate_database():
    """
    Valida a integridade do banco de dados
    
    Returns:
        dict: Relatório de validação
    """
    conn = sqlite3.connect(config.DB_NAME)
    
    # Verificar se as tabelas existem
    tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
    tables_df = pd.read_sql_query(tables_query, conn)
    
    # Contar registros
    results_count = pd.read_sql_query("SELECT COUNT(*) as count FROM pso_resultados", conn).iloc[0]['count']
    exec_count = pd.read_sql_query("SELECT COUNT(*) as count FROM pso_execucao", conn).iloc[0]['count']
    
    # Verificar experimentos únicos
    unique_experiments = pd.read_sql_query(
        "SELECT COUNT(DISTINCT num_experimento) as count FROM pso_resultados", conn
    ).iloc[0]['count']
    
    conn.close()
    
    return {
        'tables_found': tables_df['name'].tolist(),
        'pso_resultados_count': results_count,
        'pso_execucao_count': exec_count,
        'unique_experiments': unique_experiments,
        'expected_experiments': config.PSO_CONFIG['experimentos']
    }

