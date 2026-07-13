# ChessAI

Assistente de xadrez baseado em um pipeline composto por recuperação de contexto, análise de posição e geração de resposta.

## O que a aplicação faz

A aplicação combina três blocos principais:

1. **RAG (Retrieval-Augmented Generation)** - um PDF com conteúdo de xadrez é carregado, dividido em chunks e armazenado em um banco vetorial Chroma. Quando o usuário faz uma pergunta, o sistema recupera trechos relevantes para fundamentar a resposta.
2. **Análise de posição com engine** - se a mensagem contiver uma notação FEN, a aplicação usa um motor de xadrez para analisar a posição e retornar uma visão estruturada da posição.
3. **Gemini** - o modelo de linguagem gera uma resposta natural com base no contexto recuperado e na análise da posição.

## Arquitetura

Os principais componentes do projeto são:

- **PDFLoader / FileSplitter** - carregam e quebram o PDF em chunks;
- **ChromaDatabaseCreator / ChromaDatabaseRepository** - criam e consultam o banco vetorial;
- **AIGeminiService** - comunica com o modelo Gemini;
- **ChessEngineService** - valida FEN e analisa posições com o engine.

## Pré-requisitos

- Python 3.13
- Poetry
- Um arquivo de PDF chamado `chess.pdf` na raiz do projeto
- Um motor de xadrez disponível no PATH, como `stockfish`
- Uma chave da API do Google Gemini em um arquivo `.env`

## Configuração

Crie um arquivo `.env` na raiz do projeto com o conteúdo abaixo:

```env
API_KEY=sua_chave_gemini_aqui
```

Se o executável do Stockfish não estiver no PATH, ajuste `STOCKFISH_PATH` em `src/chess_ai/config.py`.

## Instalação

```bash
poetry install
```

## Execução

Para iniciar a aplicação em modo CLI:

```bash
poetry run python -m chess_ai.main
```

### Como usar

- Digite uma pergunta normal para obter uma resposta baseada no contexto do PDF.
- Use o prefixo `FEN:` para enviar uma posição e solicitar análise do engine.
- Digite `exit` ou `quit` para encerrar a sessão.

### Exemplo

```text
You: Quais são as ideias principais desta abertura? FEN: r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1
```

## Observações

A aplicação mantém o objetivo central de auxiliar o usuário em perguntas sobre xadrez, combinando recuperação de contexto, análise de posição e resposta gerada por modelo de linguagem.
