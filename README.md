# RAG Prontuário Médico

Um sistema de Retrieval-Augmented Generation (RAG) em Python projetado para processar e analisar registros de prontuários médicos. O sistema utiliza um banco de dados vetorial local (ChromaDB) e modelos locais via [Ollama](https://ollama.ai/) para garantir a privacidade dos dados do paciente.

## 🚀 Estágio Atual (Status)
**Estágio:** Prova de Conceito (PoC) / MVP.

O projeto já é funcional como uma ferramenta de linha de comando (CLI). Ele consegue:
- Gerar dados sintéticos de prontuário (simulando 9 meses de evolução de um paciente hipertenso).
- Indexar as evoluções clínicas usando *embeddings* locais (ChromaDB + SentenceTransformers).
- Interagir de forma conversacional e com memória sobre o prontuário.
- Gerar um resumo clínico estruturado do paciente de forma automatizada usando um LLM local.

## 🛠️ Tecnologias Utilizadas
- **Python 3.x**
- **Ollama** (para rodar o LLM localmente, configurado para usar o modelo `llama3` por padrão).
- **ChromaDB** (banco de dados vetorial local).
- **Sentence-Transformers** (modelo de embeddings: `paraphrase-multilingual-MiniLM-L12-v2`).
- **Requests** (para integração com a API do Ollama).

## ⚙️ Pré-requisitos
1. Python 3.8 ou superior.
2. [Ollama](https://ollama.com/) instalado e rodando em sua máquina.
3. Ter feito o pull do modelo utilizado no Ollama:
   ```bash
   ollama pull llama3
   ```

## 📦 Instalação

1. Clone ou baixe este repositório.
2. Recomenda-se criar um ambiente virtual (venv):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Como usar

Execute o arquivo principal de dentro da pasta `rag_prontuario`:

```bash
python3 main.py
```

### Argumentos da CLI
- `--chat` : Inicia a sessão de chat interativo (comportamento padrão).
- `--resumo` : Gera e exibe um resumo clínico estruturado de imediato, sem iniciar (ou além de) o chat.
- `--reindex` : Força a recriação do banco vetorial Chroma e reindexa todos os registros.
- `--modelo NOME_DO_MODELO` : Sobrescreve o modelo LLM configurado (ex: `--modelo mistral`).
- `--no-chat` : Útil em conjunto com `--resumo` se você quiser apenas gerar o resumo e encerrar a execução.

### Comandos dentro do Chat
- `/resumo` - Gera o resumo clínico estruturado do paciente.
- `/limpar` - Limpa o histórico de mensagens da conversa (útil para perguntas não relacionadas).
- `/sair` - Encerra o sistema.
- `<Qualquer outra mensagem>` - Faz uma busca vetorial no prontuário e tenta responder sua dúvida usando LLM.

## 💡 Propostas de Melhoria

Sendo um PoC, existem oportunidades de expansão:

1. **APIs e Interface Gráfica:** Mover o projeto de CLI para uma API RESTful (FastAPI) e adicionar um frontend (Streamlit, React, etc.).
2. **Modularização de Pacientes:** Atualmente os dados sintéticos são mockados no código. Adicionar suporte a múltiplos pacientes e carregamento via arquivos `.pdf`, `.txt` ou integração com sistemas HL7/FHIR.
3. **Melhorias no RAG:** 
   - Refinar a chunking strategy (tamanho de janelas de texto).
   - Implementar *Re-ranking* nos resultados do ChromaDB antes de mandar pro LLM.
4. **Resiliência e Assincronismo:** Trocar as requisições `requests` para chamadas assíncronas (`aiohttp` ou litellm), melhorando a responsividade geral e reduzindo bloqueios de thread na geração.
5. **Configuração Variável:** Mover as variáveis estáticas de `config.py` para um arquivo `.env` para facilitar o setup em ambientes diferentes.
6. **Ajuste de Imports Relativos:** Retirar os imports relativos (ex: `from . import config`) e estruturar um arquivo principal caso se queira a facilidade de executar o arquivo `main.py` diretamente de dentro da pasta.

