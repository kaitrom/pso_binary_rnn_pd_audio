import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Caminho do banco de dados
DB_PATH = 'database.db'
OUTPUT_DIR = r"graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_tables(db_path):
    conn = sqlite3.connect(db_path)
    df_res = pd.read_sql_query("SELECT * FROM pso_resultados", conn)
    df_exec = pd.read_sql_query("SELECT * FROM pso_execucao", conn)
    conn.close()
    return df_res, df_exec

def salvar_relatorio_txt(df_res, df_exec, output_path):
    cpu_medio = df_exec['uso_medio_cpu'].mean()
    memoria_media = df_exec['uso_max_memoria_mb'].mean()
    tempo_total_medio = df_exec['tempo_total_seg'].mean()
    fitness_medio = df_res['f1_score'].mean()
    melhor_fitness = df_res['f1_score'].max()
    media_geral = np.mean([cpu_medio, memoria_media, tempo_total_medio, fitness_medio, melhor_fitness])
    relatorio = (
        f"Uso médio de CPU: {cpu_medio:.2f}\n"
        f"Memória média (MB): {memoria_media:.2f}\n"
        f"Tempo total médio (segundos): {tempo_total_medio:.2f}\n"
        f"Fitness médio: {fitness_medio:.4f}\n"
        f"Melhor fitness: {melhor_fitness:.4f}\n"
        f"Média geral: {media_geral:.4f}\n"
    )
    with open(output_path, 'w') as f:
        f.write(relatorio)

def plot_entropy(df_res, experiment_num, output_dir):
    pos_cols = ['pos_camada', 'pos_n1', 'pos_n2', 'pos_n3', 'pos_lr']
    df = df_res[df_res['num_experimento'] == experiment_num]
    entropy = df.groupby('num_iteracao')[pos_cols].var().sum(axis=1)
    plt.figure(figsize=(10,5))
    plt.plot(entropy, color='purple')
    plt.title(f'Evolução da Entropia - Execução {experiment_num}')
    plt.xlabel('Iteração')
    plt.ylabel('Entropia (bits)')
    plt.grid(True)
    plt.savefig(f"{output_dir}/entropia_exec_{experiment_num}.png")
    plt.close()

def plot_fitness_convergence(df_res, experiment_num, output_dir):
    df = df_res[df_res['num_experimento'] == experiment_num]
    grouped = df.groupby('num_iteracao')
    best = grouped['f1_score'].max()
    mean = grouped['f1_score'].mean()
    worst = grouped['f1_score'].min()
    plt.figure(figsize=(10,5))
    plt.plot(best, label='Melhor Fitness', color='blue')
    plt.plot(mean, label='Fitness Médio', color='orange')
    plt.plot(worst, label='Pior Fitness', color='green')
    plt.title(f'Convergência do Fitness - Execução {experiment_num}')
    plt.xlabel('Iteração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/fitness_exec_{experiment_num}.png")
    plt.close()

def plot_histograms(df_res, experiment_num, output_dir):
    pos_cols = ['pos_camada', 'pos_n1', 'pos_n2', 'pos_n3', 'pos_lr']
    df = df_res[df_res['num_experimento'] == experiment_num]
    fig, axes = plt.subplots(1, 5, figsize=(20,4))
    for i, col in enumerate(pos_cols):
        axes[i].hist(df[col], bins=30, color='skyblue', edgecolor='black')
        mean_val = df[col].mean()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
        axes[i].set_title(f'Dimensão {i+1}')
        axes[i].set_xlabel('Valor')
        axes[i].set_ylabel('Frequência')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/histogramas_exec_{experiment_num}.png")
    plt.close()

def plot_pca_2d(df_res, experiment_num, output_dir):
    pos_cols = ['pos_camada', 'pos_n1', 'pos_n2', 'pos_n3', 'pos_lr']
    df = df_res[df_res['num_experimento'] == experiment_num]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df[pos_cols])
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.7)
    plt.title(f'PCA: Projeção 2D das 5 Dimensões - Exec {experiment_num}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% da variância)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% da variância)')
    plt.grid(True)
    plt.savefig(f"{output_dir}/pca_exec_{experiment_num}.png")
    plt.close()

def plot_pca_evolution(df_res, best_experiment_num, output_dir):
    pos_cols = ['pos_camada', 'pos_n1', 'pos_n2', 'pos_n3', 'pos_lr']
    df = df_res[df_res['num_experimento'] == best_experiment_num]
    iterations = sorted(df['num_iteracao'].unique())
    sampled = iterations[::5]
    fig, axes = plt.subplots(1, len(sampled), figsize=(4*len(sampled), 4), sharey=True)
    if len(sampled) == 1:
        axes = [axes]
    for i, it in enumerate(sampled):
        df_it = df[df['num_iteracao'] == it]
        if len(df_it) < 2:
            continue
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df_it[pos_cols])
        axes[i].scatter(X_pca[:,0], X_pca[:,1], alpha=0.7)
        axes[i].set_title(f'Iteração {it}')
        axes[i].set_xlabel('PC1')
        if i == 0:
            axes[i].set_ylabel('PC2')
        axes[i].grid(True)
    plt.suptitle(f'Evolução do PCA - Execução {best_experiment_num} (a cada 5 iterações)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_dir}/pca_evolucao_exec_{best_experiment_num}.png")
    plt.close()

# Execução principal
df_res, df_exec = read_tables(DB_PATH)
relatorio_path = os.path.join(OUTPUT_DIR, "resumo_estatisticas.txt")
salvar_relatorio_txt(df_res, df_exec, relatorio_path)

experimentos = df_res['num_experimento'].unique()
for exp_num in experimentos:
    plot_entropy(df_res, exp_num, OUTPUT_DIR)
    plot_fitness_convergence(df_res, exp_num, OUTPUT_DIR)
    plot_histograms(df_res, exp_num, OUTPUT_DIR)
    plot_pca_2d(df_res, exp_num, OUTPUT_DIR)

best_exp = df_res.groupby('num_experimento')['f1_score'].mean().idxmax()
plot_pca_evolution(df_res, best_exp, OUTPUT_DIR)
