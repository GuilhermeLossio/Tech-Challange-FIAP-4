# Nota de Atualizacao - 2026-04-17

## Tema
Integracao hibrida entre o pipeline classico de treinamento e o IBM Quantum.

## Status
Em definicao tecnica inicial.

## Resumo executivo
Foi definida como direcao inicial a avaliacao de uma arquitetura hibrida, mantendo o pipeline principal de dados, preprocessamento, engenharia de atributos e treinamento temporal no ambiente classico, com experimentos quanticos aplicados em blocos menores e controlados do fluxo.

## Contexto
O projeto atual trabalha com previsao de series temporais para ativos do setor de semicondutores, com uso de LSTM, dados historicos de mercado e enriquecimento por sentimento de noticias. Nesse cenario, a substituicao direta do modelo temporal principal por uma arquitetura quantica nao e a melhor primeira etapa.

## Decisao registrada
Adotar uma estrategia hibrida para exploracao de computacao quantica com IBM Quantum.

## Direcao tecnica proposta
- Manter em ambiente classico a ingestao de dados, limpeza, normalizacao, fusao de features e treinamento base do modelo temporal.
- Reduzir a dimensionalidade das features antes de qualquer etapa quantica, evitando enviar janelas temporais grandes diretamente para o circuito.
- Avaliar inicialmente abordagens de menor risco tecnico, como quantum kernels para classificacao de direcao de movimento ou regressao sobre retorno D+1.
- Considerar experimentos com VQR e EstimatorQNN apenas apos estabelecer um baseline classico comparavel.
- Validar primeiro em simulacao/local testing mode e depois evoluir para execucao em hardware IBM Quantum.

## Justificativa
- O pipeline atual ja possui uma separacao clara entre preprocessamento, fusao de features e inferencia.
- Series temporais financeiras com janelas extensas nao sao um bom ponto de partida para substituicao total por modelos quanticos.
- Uma abordagem hibrida permite comparar custo, qualidade preditiva e complexidade operacional com menos risco.
- O IBM Quantum pode ser explorado de forma incremental, sem comprometer o funcionamento principal do projeto.

## Proximos passos sugeridos
1. Definir o primeiro alvo experimental: classificacao de direcao ou regressao de retorno.
2. Selecionar um conjunto pequeno de features derivadas por janela.
3. Criar um baseline classico enxuto para comparacao.
4. Implementar um primeiro notebook ou modulo de prova de conceito com Qiskit.
5. Medir diferenca de desempenho, custo computacional e viabilidade pratica.

## Padrao de catalogacao adotado
- Pasta: `Docs/`
- Nome do arquivo: `YYYY-MM-DD-descricao-curta.md`
- Conteudo minimo esperado:
  - tema
  - status
  - resumo executivo
  - decisao registrada
  - justificativa
  - proximos passos

## Observacao
Esta nota inaugura o padrao de registro de atualizacoes tecnicas do projeto.
