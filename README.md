# Sistema PSO para Otimização de Hiperparâmetros - Classificação de Parkinson

Este projeto implementa um sistema modular em Python para otimização de hiperparâmetros de uma rede neural feedforward usando Particle Swarm Optimization (PSO) para classificação binária de Parkinson.

## 🎯 Características Principais

- **32 experimentos independentes** com a mesma população inicial
- **Armazenamento completo em SQLite** com dados de todas as partículas
- **Monitoramento de recursos** (CPU, memória, disco) com psutil
- **Barras de progresso** interativas com tqdm
- **Configurações centralizadas** em config.py
- **Avaliação final** com validação cruzada estratificada (k=5)
- **Arquitetura modular** para fácil manutenção e extensão

## 📁 Estrutura do Projeto

```
parkinson_pso_project/
│
├── config.py                  # Configurações centralizadas
├── data_utils.py             # Carregamento e preparação dos dados
├── model_utils.py            # Construção e treino da rede neural
├── database_utils.py         # Criação e manipulação do SQLite
├── pso_optimizer.py          # Execução do PSO com registro completo
├── evaluate_final_model.py   # Avaliação do melhor modelo
├── main.ipynb                # Notebook principal
├── test_system.py            # Script de teste do sistema
├── dataset.csv               # Dataset de Parkinson
└── README.md                 # Esta documentação
```

## 🔧 Configurações (config.py)

### Rede Neural
- **Camadas**: 1-4 camadas ocultas
- **Neurônios**: 8-128 neurônios por camada
- **Learning Rate**: 1e-5 a 1e-1
- **Épocas**: 30
- **Batch Size**: 32
- **Regularização**: BatchNormalization + Dropout(0.3)

### PSO
- **Partículas**: 20 por experimento
- **Dimensões**: 5 (n_layers, n1, n2, n3, learning_rate)
- **Iterações**: 30 por experimento
- **Experimentos**: 32 independentes
- **Parâmetros**: c1=1.5, c2=1.5, w=0.7

### Banco de Dados
- **Nome**: pso_parkinson.db
- **Tabelas**: pso_resultados, pso_execucao
- **Armazenamento**: Todos os vetores e métricas desmembradas

## 🚀 Instalação e Uso

### 1. Dependências

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn tqdm psutil
```

### 2. Teste do Sistema

```bash
python test_system.py
```

Este script executa testes completos para verificar se todos os módulos estão funcionando.

### 3. Execução Completa

#### Opção A: Notebook Jupyter
```bash
jupyter notebook main.ipynb
```

#### Opção B: Script Python
```python
import pso_optimizer
import evaluate_final_model

# Executar otimização PSO
results = pso_optimizer.run_pso_optimization()

# Avaliar melhor modelo
final_results = evaluate_final_model.evaluate_best_model()
```

### 4. Configuração para Teste Rápido

Para testes rápidos, edite `config.py`:

```python
PSO_CONFIG = {
    'n_particles': 5,      # Reduzir de 20
    'iters': 5,           # Reduzir de 30
    'experimentos': 2,    # Reduzir de 32
    # ... outros parâmetros
}
```

## 📊 Estrutura do Banco de Dados

### Tabela: pso_resultados
Registra cada partícula a cada iteração:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| num_experimento | INTEGER | Número do experimento |
| num_iteracao | INTEGER | Número da iteração |
| num_particula | INTEGER | ID da partícula |
| pos_camada, pos_n1, pos_n2, pos_n3, pos_lr | REAL | Posição da partícula |
| vel_camada, vel_n1, vel_n2, vel_n3, vel_lr | REAL | Velocidade da partícula |
| pbest_camada, pbest_n1, pbest_n2, pbest_n3, pbest_lr | REAL | Melhor posição pessoal |
| num_camadas | INTEGER | Número de camadas decodificado |
| f1_score | REAL | F1-score obtido |
| peso | REAL | Valor de fitness |
| int_best | INTEGER | 1 se é a melhor global, 0 caso contrário |

### Tabela: pso_execucao
Registra dados agregados por experimento:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| num_experimento | INTEGER | Número do experimento |
| tempo_total_seg | REAL | Tempo total em segundos |
| tempo_medio_iteracao | REAL | Tempo médio por iteração |
| tempo_medio_treino_particula | REAL | Tempo médio de treino por partícula |
| uso_medio_cpu | REAL | Uso médio de CPU (%) |
| uso_max_memoria_mb | REAL | Uso máximo de memória (MB) |
| uso_disco_mb | REAL | Uso de disco (MB) |
| total_iteracoes | INTEGER | Total de iterações executadas |

## 🧠 Arquitetura da Rede Neural

A rede neural é construída dinamicamente baseada nos parâmetros otimizados pelo PSO:

```python
# Exemplo de arquitetura otimizada
model = Sequential([
    Dense(64, activation='relu', input_shape=(22,)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(1, activation='sigmoid')  # Saída binária
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

## 📈 Métricas de Avaliação

### Durante o PSO
- **Fitness**: 1 - F1-score (minimizar)
- **Avaliação**: Train/validation split (80/20)
- **Early Stopping**: Paciência de 5 épocas

### Avaliação Final
- **Validação Cruzada**: StratifiedKFold (k=5)
- **Métricas**: F1-score, Accuracy, AUC-ROC
- **Relatórios**: Matriz de confusão, precisão, recall

## 📋 Exemplo de Vetor da Partícula

```python
# Vetor: [n_layers, n1, n2, n3, learning_rate]
position = [2.7, 64.1, 32.0, 0.0, 0.001]

# Decodificação:
n_layers = 3
neurons = [64, 32, 0]  # Apenas 2 camadas usadas
learning_rate = 0.001
```

## 📊 Dataset

- **Amostras**: 1.195
- **Features**: 22 (após remoção da coluna 'name')
- **Classes**: Binária (0=Saudável, 1=Parkinson)
- **Normalização**: MinMaxScaler
- **Balanceamento**: Dataset desbalanceado (96% Parkinson, 4% Saudável)

## 🔍 Monitoramento e Logs

O sistema monitora automaticamente:
- **Progresso**: Barras de progresso para experimentos e iterações
- **Recursos**: CPU, memória e disco em tempo real
- **Convergência**: F1-score por iteração e experimento
- **Tempo**: Duração total e por componente

## 📁 Arquivos Gerados

Após a execução completa:

- `pso_parkinson.db` - Banco SQLite com todos os resultados
- `pso_resultados.csv` - Dados detalhados das partículas
- `pso_execucao.csv` - Resumo dos experimentos
- `relatorio_avaliacao_final.txt` - Relatório completo
- `matriz_confusao.png` - Visualização da matriz de confusão

## ⚡ Performance

### Estimativas de Tempo
- **Configuração completa**: ~2-4 horas (32×20×30 = 19.200 avaliações)
- **Configuração de teste**: ~5-10 minutos (2×5×5 = 50 avaliações)

### Recursos Necessários
- **RAM**: ~2-4 GB
- **CPU**: Qualquer processador moderno
- **Disco**: ~100 MB para resultados completos

## 🛠️ Personalização

### Modificar Arquitetura da Rede
Edite `model_utils.py` na função `create_neural_network()`.

### Alterar Função de Fitness
Modifique `fitness_function()` em `model_utils.py`.

### Adicionar Novas Métricas
Estenda `evaluate_model()` em `model_utils.py`.

### Configurar Novos Parâmetros PSO
Ajuste `PSO_CONFIG` em `config.py`.

## 🐛 Solução de Problemas

### Erro de Importação TensorFlow
```bash
pip install --upgrade tensorflow
```

### Erro de Memória
Reduza o número de partículas ou experimentos em `config.py`.

### Erro de tqdm.notebook
O sistema usa automaticamente `tqdm` padrão se `tqdm.notebook` não estiver disponível.

### Dataset com Status Contínuo
O sistema converte automaticamente valores contínuos para binário usando threshold 0.5.

## 📚 Referências

- **PSO**: Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
- **Dataset**: Parkinson's Disease Classification Dataset
- **Métricas**: F1-score para classificação binária desbalanceada

## 📄 Licença

Este projeto é fornecido como está, para fins educacionais e de pesquisa.

---

**Desenvolvido com ❤️ para otimização de hiperparâmetros usando PSO**

