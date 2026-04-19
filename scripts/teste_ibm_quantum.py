"""
Script didatico para testar IBM Quantum com um experimento minimo.

Objetivo
--------
1. Rodar localmente primeiro, usando um fake backend da IBM.
2. Reaproveitar quase o mesmo codigo para rodar em hardware real.
3. Mostrar os conceitos principais do fluxo do Qiskit Runtime:
   - circuito quantico
   - transpilation / ISA circuit
   - backend
   - primitive SamplerV2
   - job mode
   - leitura de resultados

Como usar
---------
1) Instale as dependencias:
   pip install "qiskit[all]~=2.3.1" "qiskit-ibm-runtime~=0.45.1" "qiskit-aer~=0.17"

2) Teste local:
   python scripts/teste_ibm_quantum.py --modo local

3) Listar backends reais disponiveis na sua conta:
   python scripts/teste_ibm_quantum.py --modo cloud --listar-backends

4) Criar ou atualizar o arquivo .env na raiz do projeto:
   IBM_QUANTUM_API_TOKEN=SEU_TOKEN
   IBM_QUANTUM_INSTANCE=SUA_INSTANCE_OU_CRN

5) Rodar em hardware real:
   python scripts/teste_ibm_quantum.py --modo cloud --backend ibm_brisbane

Observacoes importantes
-----------------------
- Este script usa "job mode", que e compativel com Open Plan.
- Nao usamos Session porque a documentacao oficial informa que Open Plan
  nao aceita session jobs.
- Em modo local, o script usa FakeManilaV2 para simular um backend IBM.
- O experimento escolhido e um Bell state, porque ele e pequeno e mostra
  entanglement de forma clara: o esperado e aparecerem mais os estados
  "00" e "11".
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
except ImportError as exc:  # pragma: no cover - mensagem amigavel para setup manual
    raise SystemExit(
        "Dependencias ausentes. Instale com:\n"
        'pip install "qiskit[all]~=2.3.1" "qiskit-ibm-runtime~=0.45.1" "qiskit-aer~=0.17"\n'
        f"\nDetalhe tecnico: {exc}"
    )


DEFAULT_SHOTS = 1024
DEFAULT_OPT_LEVEL = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass
class ExecucaoConfig:
    modo: str
    backend_nome: str | None
    shots: int
    optimization_level: int
    listar_backends: bool


def limpar_valor_env(valor: str) -> str:
    """
    Remove aspas ao redor do valor quando o .env usa strings entre aspas.
    """
    valor = valor.strip()
    if len(valor) >= 2 and valor[0] == valor[-1] and valor[0] in {"'", '"'}:
        return valor[1:-1]
    return valor


def carregar_env_arquivo(caminho_env: Path = DEFAULT_ENV_PATH) -> bool:
    """
    Carrega um .env simples usando apenas a biblioteca padrao.

    Regras:
    - ignora comentarios e linhas vazias
    - aceita prefixo opcional "export "
    - nao sobrescreve variaveis ja definidas no ambiente
    """
    if not caminho_env.exists():
        return False

    for linha_bruta in caminho_env.read_text(encoding="utf-8").splitlines():
        linha = linha_bruta.strip()
        if not linha or linha.startswith("#"):
            continue

        if linha.startswith("export "):
            linha = linha[len("export ") :].strip()

        if "=" not in linha:
            continue

        chave, valor = linha.split("=", 1)
        chave = chave.strip()
        valor = limpar_valor_env(valor)

        if not chave:
            continue

        os.environ.setdefault(chave, valor)

    return True


def parse_args() -> ExecucaoConfig:
    parser = argparse.ArgumentParser(
        description="Teste minimo e comentado para IBM Quantum."
    )
    parser.add_argument(
        "--modo",
        choices=("local", "cloud"),
        default="local",
        help="local usa fake backend; cloud usa IBM Quantum Runtime.",
    )
    parser.add_argument(
        "--backend",
        dest="backend_nome",
        default=None,
        help=(
            "Nome do backend real no modo cloud. "
            "Se omitido, o script tenta escolher o least_busy automaticamente."
        ),
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help="Numero de amostragens da primitive SamplerV2.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=DEFAULT_OPT_LEVEL,
        choices=(0, 1, 2, 3),
        help="Nivel de otimizacao da transpilation.",
    )
    parser.add_argument(
        "--listar-backends",
        action="store_true",
        help="No modo cloud, lista alguns backends reais acessiveis e encerra.",
    )
    args = parser.parse_args()
    return ExecucaoConfig(
        modo=args.modo,
        backend_nome=args.backend_nome,
        shots=args.shots,
        optimization_level=args.optimization_level,
        listar_backends=args.listar_backends,
    )


def construir_circuito_bell() -> QuantumCircuit:
    """
    Cria um circuito de Bell.

    Conceito:
    - H no qubit 0 cria superposicao.
    - CX entre q0 e q1 cria entanglement.
    - measure_all adiciona registradores classicos e mede os qubits.
    """
    circuito = QuantumCircuit(2)
    circuito.h(0)
    circuito.cx(0, 1)
    circuito.measure_all()
    return circuito


def extrair_counts(pub_result: Any) -> dict[str, int]:
    """
    Tenta encontrar o BitArray de medicao e converte para contagem por bitstring.

    Em muitos exemplos da documentacao, quando usamos measure_all(),
    o nome do registrador classico padrao e "meas". Ainda assim, este helper
    percorre os campos disponiveis para ficar mais robusto.
    """
    data = pub_result.data

    if hasattr(data, "meas") and hasattr(data.meas, "get_counts"):
        return dict(data.meas.get_counts())

    for attr_name in dir(data):
        if attr_name.startswith("_"):
            continue
        attr_value = getattr(data, attr_name)
        if hasattr(attr_value, "get_counts"):
            return dict(attr_value.get_counts())

    raise RuntimeError(
        "Nao foi possivel localizar um registrador classico com get_counts() no resultado."
    )


def imprimir_counts(counts: dict[str, int], shots: int) -> None:
    print("\nResultados (counts):")
    for bitstring, total in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        percentual = (total / shots) * 100
        print(f"  {bitstring}: {total} shots ({percentual:.2f}%)")


def construir_servico_cloud() -> QiskitRuntimeService:
    """
    Inicializa o service do IBM Quantum Runtime.

    Regras adotadas aqui:
    - Primeiro o script tenta ler credenciais do arquivo .env.
    - Se houver token no ambiente, o script instancia explicitamente o service.
    - Se nao houver token no ambiente, tenta usar credenciais previamente salvas.

    Variaveis de ambiente esperadas por este script:
    - IBM_QUANTUM_API_TOKEN
    - IBM_QUANTUM_INSTANCE (recomendado pela documentacao oficial)
    """
    carregar_env_arquivo()

    token = os.getenv("IBM_QUANTUM_API_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")

    if token:
        kwargs: dict[str, Any] = {
            "channel": "ibm_quantum_platform",
            "token": token,
        }
        if instance:
            kwargs["instance"] = instance
        return QiskitRuntimeService(**kwargs)

    return QiskitRuntimeService()


def escolher_backend_cloud(
    service: QiskitRuntimeService,
    backend_nome: str | None,
):
    """
    Escolhe o backend real.

    Se o usuario informar --backend, respeitamos o nome.
    Caso contrario, usamos least_busy para buscar um backend operacional.
    """
    if backend_nome:
        return service.backend(backend_nome)

    return service.least_busy(operational=True, simulator=False, min_num_qubits=2)


def listar_backends_cloud(service: QiskitRuntimeService) -> None:
    backends = service.backends(operational=True, simulator=False)
    if not backends:
        print("Nenhum backend real encontrado para a conta atual.")
        return

    print("Backends reais acessiveis:")
    for backend in backends[:15]:
        nome = getattr(backend, "name", str(backend))
        qubits = getattr(backend, "num_qubits", "desconhecido")
        status = "operational"
        print(f"  - {nome} | qubits={qubits} | status={status}")


def executar_local(config: ExecucaoConfig) -> None:
    print("Modo selecionado: local")
    print("Contexto: usando FakeManilaV2 para validar o fluxo antes da QPU real.")

    circuito = construir_circuito_bell()
    backend = FakeManilaV2()

    print(f"Backend local: {backend.name}")
    print("Etapa: transpilation para adequar o circuito ao backend escolhido.")
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=config.optimization_level,
    )
    isa_circuito = pass_manager.run(circuito)

    print("Etapa: execucao com SamplerV2 em job mode.")
    sampler = Sampler(
        mode=backend,
        options={"simulator": {"seed_simulator": 42}},
    )
    job = sampler.run([isa_circuito], shots=config.shots)
    result = job.result()
    counts = extrair_counts(result[0])

    print("Leitura final: os estados dominantes devem tender a 00 e 11.")
    imprimir_counts(counts, config.shots)


def executar_cloud(config: ExecucaoConfig) -> None:
    print("Modo selecionado: cloud")
    print("Contexto: usando IBM Quantum Runtime com SamplerV2 em job mode.")
    if DEFAULT_ENV_PATH.exists():
        print(f"Arquivo .env detectado: {DEFAULT_ENV_PATH}")
    else:
        print("Arquivo .env nao encontrado; o script tentara usar credenciais ja salvas.")

    try:
        service = construir_servico_cloud()
    except Exception as exc:
        raise SystemExit(
            "Nao foi possivel autenticar no IBM Quantum Runtime.\n"
            "Defina IBM_QUANTUM_API_TOKEN e, de preferencia, IBM_QUANTUM_INSTANCE no .env,\n"
            "ou salve sua conta com QiskitRuntimeService.save_account(...).\n"
            f"\nDetalhe tecnico: {exc}"
        )

    if config.listar_backends:
        listar_backends_cloud(service)
        return

    backend = escolher_backend_cloud(service, config.backend_nome)
    backend_nome = getattr(backend, "name", str(backend))

    print(f"Backend real escolhido: {backend_nome}")
    print("Etapa: montar o Bell circuit.")
    circuito = construir_circuito_bell()

    print("Etapa: transpilation local para o ISA circuit do backend real.")
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=config.optimization_level,
    )
    isa_circuito = pass_manager.run(circuito)

    print("Etapa: envio do job para a IBM Quantum Platform.")
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuito], shots=config.shots)
    print(f"Job ID: {job.job_id()}")
    print("Aguardando resultado remoto...")

    result = job.result()
    counts = extrair_counts(result[0])

    print("Resultado recebido da plataforma.")
    imprimir_counts(counts, config.shots)


def main() -> int:
    config = parse_args()

    if config.modo == "local":
        executar_local(config)
        return 0

    executar_cloud(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
