"""
Script de teste para verificar se todos os módulos do sistema PSO estão funcionando
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Testa se todas as importações estão funcionando"""
    print("🔍 Testando importações...")
    
    try:
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        import sklearn
        import psutil
        import tqdm
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Bibliotecas externas importadas com sucesso")
    except Exception as e:
        print(f"❌ Erro nas bibliotecas externas: {e}")
        return False
    
    try:
        import config
        import data_utils
        import model_utils
        import database_utils
        import pso_optimizer
        import evaluate_final_model
        print("✅ Módulos do projeto importados com sucesso")
    except Exception as e:
        print(f"❌ Erro nos módulos do projeto: {e}")
        return False
    
    return True

def test_data_loading():
    """Testa carregamento e preparação dos dados"""
    print("\n📊 Testando carregamento de dados...")
    
    try:
        import data_utils
        
        # Carregar dados
        df = data_utils.load_parkinson_data()
        print(f"✅ Dataset carregado: {df.shape}")
        
        # Validar dados
        data_utils.validate_data(df)
        print("✅ Dados validados")
        
        # Preparar dados
        X_data, y_data, scaler = data_utils.prepare_data_for_cv(df)
        print(f"✅ Dados preparados: X{X_data.shape}, y{y_data.shape}")
        
        return True, X_data, y_data
        
    except Exception as e:
        print(f"❌ Erro no carregamento de dados: {e}")
        traceback.print_exc()
        return False, None, None

def test_neural_network(X_data, y_data):
    """Testa criação e treino da rede neural"""
    print("\n🧠 Testando rede neural...")
    
    try:
        import model_utils
        from sklearn.model_selection import train_test_split
        
        # Dividir dados
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )
        
        # Parâmetros de teste
        test_params = {
            'n_layers': 2,
            'neurons': [32, 16],
            'learning_rate': 0.001
        }
        
        # Criar modelo
        model = model_utils.create_neural_network(X_data.shape[1], test_params)
        print("✅ Modelo criado")
        
        # Treinar por poucas épocas
        history = model_utils.train_neural_network(
            model, X_train, y_train, X_val, y_val, epochs=3, verbose=0
        )
        print("✅ Modelo treinado")
        
        # Avaliar
        metrics = model_utils.evaluate_model(model, X_val, y_val)
        print(f"✅ Modelo avaliado - F1: {metrics['f1_score']:.4f}")
        
        # Limpar memória
        del model
        import tensorflow as tf
        tf.keras.backend.clear_session()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na rede neural: {e}")
        traceback.print_exc()
        return False

def test_database():
    """Testa criação e operações do banco de dados"""
    print("\n💾 Testando banco de dados...")
    
    try:
        import database_utils
        import os
        
        # Criar banco
        db_path = database_utils.create_database()
        print("✅ Banco de dados criado")
        
        # Testar inserção de dados fictícios
        test_particles = [{
            'particle_id': 0,
            'position': [2.0, 32.0, 16.0, 8.0, 0.001],
            'velocity': [0.1, 1.0, 0.5, 0.2, 0.0001],
            'pbest_position': [2.0, 32.0, 16.0, 8.0, 0.001],
            'num_layers': 2,
            'f1_score': 0.85,
            'fitness': 0.15,
            'is_gbest': True
        }]
        
        database_utils.insert_pso_results_batch(0, 0, test_particles)
        print("✅ Dados inseridos no banco")
        
        # Testar resumo do experimento
        summary_data = {
            'tempo_total_seg': 100.0,
            'tempo_medio_iteracao': 3.33,
            'tempo_medio_treino_particula': 0.5,
            'uso_medio_cpu': 50.0,
            'uso_max_memoria_mb': 512.0,
            'uso_disco_mb': 1024.0,
            'total_iteracoes': 30
        }
        
        database_utils.insert_experiment_summary(0, summary_data)
        print("✅ Resumo do experimento inserido")
        
        # Validar banco
        validation = database_utils.validate_database()
        print(f"✅ Banco validado - {validation['pso_resultados_count']} registros")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no banco de dados: {e}")
        traceback.print_exc()
        return False

def test_pso_mini():
    """Testa uma versão mini do PSO (1 experimento, 2 partículas, 2 iterações)"""
    print("\n🔄 Testando PSO mini...")
    
    try:
        # Modificar configurações temporariamente para teste rápido
        import config
        original_config = {
            'experimentos': config.PSO_CONFIG['experimentos'],
            'n_particles': config.PSO_CONFIG['n_particles'],
            'iters': config.PSO_CONFIG['iters']
        }
        
        # Configuração mini para teste
        config.PSO_CONFIG['experimentos'] = 1
        config.PSO_CONFIG['n_particles'] = 2
        config.PSO_CONFIG['iters'] = 2
        
        print(f"⚙️ Configuração mini: {config.PSO_CONFIG['experimentos']} exp, {config.PSO_CONFIG['n_particles']} part, {config.PSO_CONFIG['iters']} iter")
        
        # Carregar dados
        import data_utils
        df = data_utils.load_parkinson_data()
        X_data, y_data, scaler = data_utils.prepare_data_for_cv(df)
        
        # Criar otimizador
        import pso_optimizer
        optimizer = pso_optimizer.PSOOptimizer(X_data, y_data)
        
        # Executar um experimento
        result = optimizer.run_single_experiment(0, random_state=42)
        print(f"✅ PSO mini executado - F1: {result['best_f1_score']:.4f}")
        
        # Restaurar configurações originais
        config.PSO_CONFIG.update(original_config)
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no PSO mini: {e}")
        traceback.print_exc()
        
        # Restaurar configurações em caso de erro
        try:
            config.PSO_CONFIG.update(original_config)
        except:
            pass
        
        return False

def main():
    """Função principal de teste"""
    print("🧪 INICIANDO TESTES DO SISTEMA PSO")
    print("=" * 50)
    
    # Lista de testes
    tests = [
        ("Importações", test_imports),
        ("Carregamento de dados", test_data_loading),
        ("Rede neural", None),  # Será executado com dados
        ("Banco de dados", test_database),
        ("PSO mini", test_pso_mini)
    ]
    
    results = []
    X_data, y_data = None, None
    
    for test_name, test_func in tests:
        if test_name == "Carregamento de dados":
            success, X_data, y_data = test_func()
            results.append((test_name, success))
        elif test_name == "Rede neural":
            if X_data is not None and y_data is not None:
                success = test_neural_network(X_data, y_data)
                results.append((test_name, success))
            else:
                results.append((test_name, False))
        else:
            success = test_func()
            results.append((test_name, success))
    
    # Resumo dos testes
    print("\n" + "=" * 50)
    print("📋 RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{test_name:20s} - {status}")
        if success:
            passed += 1
    
    print("=" * 50)
    print(f"📊 Resultado: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema funcionando corretamente.")
        return True
    else:
        print("⚠️ Alguns testes falharam. Verifique os erros acima.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

