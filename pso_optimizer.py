"""
Otimizador PSO principal para otimização de hiperparâmetros da rede neural
"""

import numpy as np
import time
import psutil
import os
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
import data_utils
import model_utils
import database_utils

class PSOOptimizer:
    """
    Classe principal para otimização PSO
    """
    
    def __init__(self, X_data, y_data):
        """
        Inicializa o otimizador PSO
        
        Args:
            X_data: Features do dataset
            y_data: Labels do dataset
        """
        self.X_data = X_data
        self.y_data = y_data
        self.input_dim = X_data.shape[1]
        
        # Configurações PSO
        self.n_particles = config.PSO_CONFIG['n_particles']
        self.dimensions = config.PSO_CONFIG['dimensions']
        self.max_iters = config.PSO_CONFIG['iters']
        self.n_experiments = config.PSO_CONFIG['experimentos']
        
        # Parâmetros PSO
        self.c1 = config.PSO_CONFIG['options']['c1']
        self.c2 = config.PSO_CONFIG['options']['c2']
        self.w = config.PSO_CONFIG['options']['w']
        
        # Limites
        self.bounds_lower = np.array(config.PSO_CONFIG['bounds']['lower'])
        self.bounds_upper = np.array(config.PSO_CONFIG['bounds']['upper'])
        
        # Inicializar banco de dados
        database_utils.create_database()
        
        # Preparar dados de treino/validação
        self._prepare_train_val_data()
        
    def _prepare_train_val_data(self):
        """
        Prepara dados de treino e validação
        """
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_data, self.y_data,
            test_size=config.NN_CONFIG['validation_split'],
            random_state=config.NN_CONFIG['random_state'],
            stratify=self.y_data
        )
        
    def _initialize_swarm(self, random_state=None):
        """
        Inicializa o enxame de partículas
        
        Args:
            random_state: Seed para reprodutibilidade
            
        Returns:
            dict: Dados do enxame inicializado
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Inicializar posições aleatórias dentro dos limites
        positions = np.random.uniform(
            self.bounds_lower, 
            self.bounds_upper, 
            (self.n_particles, self.dimensions)
        )
        
        # Inicializar velocidades
        velocities = np.random.uniform(
            -1, 1, 
            (self.n_particles, self.dimensions)
        )
        
        # Inicializar melhores posições pessoais
        pbest_positions = positions.copy()
        pbest_fitness = np.full(self.n_particles, float('inf'))
        
        # Melhor posição global
        gbest_position = None
        gbest_fitness = float('inf')
        gbest_particle_id = -1
        
        return {
            'positions': positions,
            'velocities': velocities,
            'pbest_positions': pbest_positions,
            'pbest_fitness': pbest_fitness,
            'gbest_position': gbest_position,
            'gbest_fitness': gbest_fitness,
            'gbest_particle_id': gbest_particle_id
        }
    
    def _evaluate_particle(self, position):
        """
        Avalia uma partícula (calcula fitness)
        
        Args:
            position: Posição da partícula
            
        Returns:
            tuple: (fitness, f1_score)
        """
        try:
            fitness = model_utils.fitness_function(
                position, 
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                self.input_dim
            )
            f1_score = 1.0 - fitness  # Converter de volta para F1-score
            return fitness, f1_score
        except Exception as e:
            print(f"Erro na avaliação da partícula: {str(e)}")
            print(traceback.print_exc())
            return 1.0, 0.0  # Pior fitness possível
    
    def _update_velocity(self, velocity, position, pbest_pos, gbest_pos):
        """
        Atualiza velocidade da partícula
        
        Args:
            velocity: Velocidade atual
            position: Posição atual
            pbest_pos: Melhor posição pessoal
            gbest_pos: Melhor posição global
            
        Returns:
            np.array: Nova velocidade
        """
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)
        
        cognitive = self.c1 * r1 * (pbest_pos - position)
        social = self.c2 * r2 * (gbest_pos - position)
        
        new_velocity = self.w * velocity + cognitive + social
        
        # Limitar velocidade
        max_velocity = 0.1 * (self.bounds_upper - self.bounds_lower)
        new_velocity = np.clip(new_velocity, -max_velocity, max_velocity)
        
        return new_velocity
    
    def _update_position(self, position, velocity):
        """
        Atualiza posição da partícula
        
        Args:
            position: Posição atual
            velocity: Velocidade atual
            
        Returns:
            np.array: Nova posição
        """
        new_position = position + velocity
        
        # Garantir que a posição está dentro dos limites
        new_position = np.clip(new_position, self.bounds_lower, self.bounds_upper)
        
        return new_position
    
    def _collect_particle_data(self, swarm, iteration, fitness_scores, f1_scores):
        """
        Coleta dados das partículas para armazenamento
        
        Args:
            swarm: Dados do enxame
            iteration: Número da iteração
            fitness_scores: Scores de fitness
            f1_scores: Scores F1
            
        Returns:
            list: Lista com dados de cada partícula
        """
        particles_data = []
        
        for i in range(self.n_particles):
            # Decodificar parâmetros da rede neural
            params = model_utils.decode_particle_position(swarm['positions'][i])
            
            particle_data = {
                'particle_id': i,
                'position': swarm['positions'][i].tolist(),
                'velocity': swarm['velocities'][i].tolist(),
                'pbest_position': swarm['pbest_positions'][i].tolist(),
                'num_layers': params['n_layers'],
                'f1_score': f1_scores[i],
                'fitness': fitness_scores[i],
                'is_gbest': (i == swarm['gbest_particle_id'])
            }
            
            particles_data.append(particle_data)
        
        return particles_data
    
    def _monitor_resources(self):
        """
        Monitora uso de recursos do sistema
        
        Returns:
            dict: Dados de recursos
        """
        process = psutil.Process(os.getpid())
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'disk_usage_mb': psutil.disk_usage('.').used / 1024 / 1024
        }
    
    def run_single_experiment(self, experiment_num, random_state=None):
        """
        Executa um único experimento PSO
        
        Args:
            experiment_num: Número do experimento
            random_state: Seed para reprodutibilidade
            
        Returns:
            dict: Resultados do experimento
        """
        start_time = time.time()
        
        # Inicializar enxame
        swarm = self._initialize_swarm(random_state)
        
        # Listas para monitoramento
        iteration_times = []
        particle_times = []
        resource_data = []
        
        # Barra de progresso para iterações
        pbar_iter = tqdm(
            range(self.max_iters), 
            desc=f"Exp {experiment_num+1:2d} - Iterações",
            leave=False
        )
        
        for iteration in pbar_iter:
            iter_start_time = time.time()
            
            # Avaliar todas as partículas
            fitness_scores = []
            f1_scores = []
            
            for i in tqdm(range(self.n_particles), desc="Particles"):
                particle_start_time = time.time()
                
                fitness, f1_score = self._evaluate_particle(swarm['positions'][i])
                fitness_scores.append(fitness)
                f1_scores.append(f1_score)
                
                particle_times.append(time.time() - particle_start_time)
                
                # Atualizar melhor posição pessoal
                if fitness < swarm['pbest_fitness'][i]:
                    swarm['pbest_fitness'][i] = fitness
                    swarm['pbest_positions'][i] = swarm['positions'][i].copy()
                
                # Atualizar melhor posição global
                if fitness < swarm['gbest_fitness']:
                    swarm['gbest_fitness'] = fitness
                    swarm['gbest_position'] = swarm['positions'][i].copy()
                    swarm['gbest_particle_id'] = i
            
            # Atualizar velocidades e posições
            for i in range(self.n_particles):
                swarm['velocities'][i] = self._update_velocity(
                    swarm['velocities'][i],
                    swarm['positions'][i],
                    swarm['pbest_positions'][i],
                    swarm['gbest_position']
                )
                
                swarm['positions'][i] = self._update_position(
                    swarm['positions'][i],
                    swarm['velocities'][i]
                )
            
            # Coletar dados das partículas
            particles_data = self._collect_particle_data(
                swarm, iteration, fitness_scores, f1_scores
            )
            
            # Salvar dados no banco
            database_utils.insert_pso_results_batch(
                experiment_num, iteration, particles_data
            )
            
            # Monitorar recursos
            resources = self._monitor_resources()
            resource_data.append(resources)
            
            iteration_times.append(time.time() - iter_start_time)
            
            # Atualizar barra de progresso
            best_f1 = max(f1_scores)
            pbar_iter.set_postfix({
                'Best F1': f'{best_f1:.4f}',
                'Avg F1': f'{np.mean(f1_scores):.4f}'
            })
        
        pbar_iter.close()
        
        # Calcular estatísticas do experimento
        total_time = time.time() - start_time
        avg_iteration_time = np.mean(iteration_times)
        avg_particle_time = np.mean(particle_times)
        avg_cpu = np.mean([r['cpu_percent'] for r in resource_data])
        max_memory = max([r['memory_mb'] for r in resource_data])
        avg_disk = np.mean([r['disk_usage_mb'] for r in resource_data])
        
        # Salvar resumo do experimento
        summary_data = {
            'tempo_total_seg': total_time,
            'tempo_medio_iteracao': avg_iteration_time,
            'tempo_medio_treino_particula': avg_particle_time,
            'uso_medio_cpu': avg_cpu,
            'uso_max_memoria_mb': max_memory,
            'uso_disco_mb': avg_disk,
            'total_iteracoes': self.max_iters
        }
        
        database_utils.insert_experiment_summary(experiment_num, summary_data)
        
        return {
            'experiment_num': experiment_num,
            'best_fitness': swarm['gbest_fitness'],
            'best_f1_score': 1.0 - swarm['gbest_fitness'],
            'best_position': swarm['gbest_position'],
            'total_time': total_time,
            'summary': summary_data
        }
    
    def run_all_experiments(self):
        """
        Executa todos os experimentos PSO
        
        Returns:
            dict: Resultados de todos os experimentos
        """
        print(f"Iniciando {self.n_experiments} experimentos PSO...")
        print(f"Configuração: {self.n_particles} partículas, {self.max_iters} iterações")
        
        # Barra de progresso principal
        pbar_main = tqdm(
            range(self.n_experiments), 
            desc="Experimentos",
            position=0
        )
        
        all_results = []
        
        for exp_num in pbar_main:
            # Usar seed diferente para cada experimento
            random_state = config.PSO_CONFIG['random_state'] + exp_num
            
            result = self.run_single_experiment(exp_num, random_state)
            all_results.append(result)
            
            # Atualizar barra de progresso principal
            pbar_main.set_postfix({
                'Best F1': f'{result["best_f1_score"]:.4f}',
                'Time': f'{result["total_time"]:.1f}s'
            })
        
        pbar_main.close()
        
        # Encontrar melhor resultado geral
        best_experiment = max(all_results, key=lambda x: x['best_f1_score'])
        
        print(f"\n✅ Todos os experimentos concluídos!")
        print(f"Melhor F1-score: {best_experiment['best_f1_score']:.4f}")
        print(f"Experimento: {best_experiment['experiment_num'] + 1}")
        
        return {
            'all_results': all_results,
            'best_experiment': best_experiment,
            'best_particle_data': database_utils.get_best_particle_overall()
        }

def run_pso_optimization():
    """
    Função principal para executar a otimização PSO
    
    Returns:
        dict: Resultados da otimização
    """
    print("🚀 Iniciando otimização PSO para classificação de Parkinson")
    
    # Carregar e preparar dados
    print("📊 Carregando dados...")
    df = data_utils.load_parkinson_data()
    data_utils.validate_data(df)
    
    X_data, y_data, scaler = data_utils.prepare_data_for_cv(df)
    
    print(f"Dataset: {X_data.shape[0]} amostras, {X_data.shape[1]} features")
    print(f"Distribuição de classes: {np.bincount(y_data.astype(int))}")
    
    # Criar otimizador
    optimizer = PSOOptimizer(X_data, y_data)
    
    # Executar otimização
    results = optimizer.run_all_experiments()
    
    # Exportar resultados
    print("\n💾 Exportando resultados...")
    database_utils.export_results_to_csv()
    
    # Estatísticas finais
    stats = database_utils.get_experiment_statistics()
    print(f"\n📈 Estatísticas finais:")
    print(f"F1-score médio: {stats.get('avg_f1_score', 0):.4f}")
    print(f"Melhor F1-score: {stats.get('best_f1_score', 0):.4f}")
    print(f"Tempo médio por experimento: {stats.get('avg_total_time', 0):.1f}s")
    
    return results

if __name__ == "__main__":
    results = run_pso_optimization()

