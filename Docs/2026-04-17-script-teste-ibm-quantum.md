# Nota de Atualizacao - 2026-04-17

## Tema
Criacao de script didatico para teste inicial com IBM Quantum.

## Status
Concluido.

## Resumo executivo
Foi adicionado um script introdutorio para validar o fluxo de execucao com Qiskit Runtime, primeiro em modo local e depois em modo cloud. O objetivo e permitir aprendizado progressivo do processo tecnico sem depender, no primeiro contato, da execucao imediata em hardware real.

## Decisao registrada
Adotar um script de smoke test com circuito de Bell e `SamplerV2`, priorizando simplicidade, clareza conceitual e compatibilidade com `job mode`.

## Artefatos adicionados
- `scripts/teste_ibm_quantum.py`

## Justificativa
- O circuito de Bell e pequeno, classico para validacao inicial e facil de interpretar.
- O modo local reduz friccao de setup e ajuda a validar o fluxo antes da autenticacao remota.
- O uso de `SamplerV2` acompanha o modelo atual da documentacao oficial do IBM Quantum Runtime.
- O uso de `job mode` evita conflito com a restricao do Open Plan sobre `session jobs`.

## Conteudos didaticos incluidos no script
- diferenca entre modo local e modo cloud
- objetivo da transpilation
- papel do backend
- uso de `SamplerV2`
- leitura de `counts`
- autenticacao via token e instance

## Proximos passos sugeridos
1. Executar o script em modo local e validar o resultado esperado.
2. Configurar token e instance da IBM Quantum Platform.
3. Listar backends acessiveis na conta.
4. Executar o mesmo experimento em hardware real.
5. Evoluir do Bell test para um experimento ligado a features do projeto.

## Observacao
Este registro complementa a nota anterior sobre integracao hibrida e documenta o primeiro artefato pratico dessa direcao.
