
    # Relatório de Teste de Stress: Streaming MAD (Numba JIT)
    - **Amostras Processadas:** 1,000,000
    - **Tempo Total:** 13.4905 segundos
    - **Throughput:** 74,126.20 amostras/segundo
    - **Latência Média por Evento:** 13.4905 microsegundos
    - **Anomalias Corretamente Barradas:** 1155
    
    **Veredito:** O motor streaming MAD atinge performance em nível de microssegundos devido ao Numba JIT, o que atende 100% aos requisitos de um analisador de custo "shift-left" em tempo real sem degradar as esteiras.
    