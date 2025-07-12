# 🤖 Guia dos Agentes e Projetos IA

Este guia documenta todos os projetos de IA, agentes e ferramentas disponíveis na pasta **AGENTES**. Cada projeto tem sua própria finalidade e forma de uso.

---

## 📋 Índice

- [🧠 Modelos de IA](#-modelos-de-ia)
- [🔧 Servidores MCP](#-servidores-mcp)
- [💬 Interfaces de Chat](#-interfaces-de-chat)
- [🛠️ Ferramentas de Desenvolvimento](#️-ferramentas-de-desenvolvimento)
- [📚 Tutoriais e Exemplos](#-tutoriais-e-exemplos)
- [🎯 Projetos Especializados](#-projetos-especializados)
- [📖 Recursos e Documentação](#-recursos-e-documentação)

---

## 🧠 Modelos de IA

### 📄 Llama3_2_(1B_and_3B)_Conversational.ipynb
**Notebook Jupyter para fine-tuning de modelos Llama 3.2**

- **Finalidade**: Tutorial completo para treinar modelos Llama 3.2 (1B e 3B parâmetros) usando Unsloth
- **Como usar**:
  1. Execute no Google Colab com GPU Tesla T4 gratuita
  2. Pressione "Runtime" → "Run all"
  3. O notebook inclui instalação, preparação de dados, treinamento e inferência
- **Funcionalidades**:
  - Fine-tuning com LoRA adapters
  - Formato de conversação Llama-3.1
  - Dataset FineTome-100k
  - Quantização 4-bit para economia de memória

### 🧮 modelo-bitnet-b1.58-2B-4T/
**Modelo BitNet B1.58 da Microsoft - LLM nativo de 1-bit**

- **Finalidade**: Modelo de linguagem ultra-eficiente com pesos de 1.58-bit
- **Características**:
  - 2 bilhões de parâmetros
  - Treinado em 4 trilhões de tokens
  - Memória não-embedding: apenas 0.4GB
  - Latência CPU: 29ms por token
- **Como usar**:
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model_id = "microsoft/bitnet-b1.58-2B-4T"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
  ```
- **⚠️ Importante**: Para máxima eficiência, use `bitnet.cpp` em vez do transformers

### 🦙 modelo-llama-cpp-python/
**Bindings Python para llama.cpp**

- **Finalidade**: Interface Python para executar modelos Llama localmente
- **Como usar**:
  ```bash
  pip install llama-cpp-python
  ```
  ```python
  from llama_cpp import Llama
  llm = Llama(model_path="./models/7B/ggml-model.bin")
  output = llm("Sua pergunta aqui", max_tokens=32)
  ```
- **Funcionalidades**:
  - API compatível com OpenAI
  - Servidor web integrado
  - Suporte LangChain
  - Acesso completo à API C do llama.cpp

---

## 🔧 Servidores MCP

### 🧊 cryo-mcp/
**Servidor MCP para extração de dados blockchain**

- **Finalidade**: Acessar dados da blockchain Ethereum via protocolo MCP
- **Como usar**:
  ```bash
  # Instalar
  uv tool install cryo-mcp
  
  # Executar
  uvx cryo-mcp -r <ETH_RPC_URL>
  ```
- **Funcionalidades principais**:
  - `list_datasets()` - Lista datasets disponíveis
  - `query_dataset()` - Consulta dados blockchain
  - `get_latest_ethereum_block()` - Último bloco
  - Filtros por contrato, intervalo de blocos
  - Exportação JSON/CSV

### 🐙 github-mcp-server/
**Servidor MCP para automação GitHub**

- **Finalidade**: Integração completa com APIs do GitHub via MCP
- **Como usar**:
  ```bash
  # Via Docker
  docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server
  ```
- **Funcionalidades**:
  - **Issues**: criar, listar, comentar, atualizar
  - **Pull Requests**: gerenciar, revisar, mesclar
  - **Repositórios**: criar, buscar, gerenciar arquivos
  - **Busca**: código, usuários, repositórios
  - **Code Scanning**: alertas de segurança
- **Requisitos**: Token de acesso pessoal do GitHub

---

## 💬 Interfaces de Chat

### 🤖 chatbot-deepseek-interface/
**Interface web para modelo R1 Deepseek**

- **Finalidade**: Interface web completa para interagir com modelos Deepseek
- **Como instalar**:
  ```bash
  # Dependências Python
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install transformers python-bridge
  
  # Dependências Node.js
  cd interface1
  pnpm install
  pnpm build
  ```
- **Como usar**:
  ```bash
  cd interface1/apps/web
  pnpm dev
  # Acesse http://localhost:3000
  ```
- **Interfaces disponíveis**:
  - **Interface 1**: Design minimalista (`/interface1`)
  - **Interface 2**: Layout com histórico (`/interface2`)
- **⚠️ Requisitos**: 8GB RAM, modelo executa em CPU

---

## 🛠️ Ferramentas de Desenvolvimento

### ⚡ codex/
**CLI de programação com IA (fork melhorado)**

- **Finalidade**: Agente de programação que roda no terminal
- **Como instalar**:
  ```bash
  npm install -g @ymichael/codex
  export OPENAI_API_KEY="sua-chave-aqui"
  ```
- **Como usar**:
  ```bash
  # Modo interativo
  codex
  
  # Com prompt direto
  codex "explique este código"
  
  # Modo automático
  codex --approval-mode full-auto "crie um app todo"
  ```
- **Funcionalidades**:
  - Múltiplos provedores: OpenAI, Gemini, OpenRouter, Ollama
  - Sandboxing de segurança
  - Aprovação automática configurável
  - Integração com Git
  - Suporte multimodal

### 🔗 openai-mcp-client/
**Cliente MCP para OpenAI**

- **Finalidade**: Exemplo de integração MCP com API OpenAI
- **Como usar**:
  ```bash
  # Instalar Deno v2
  deno install
  
  # Configurar .env
  cp .env.example .env
  # Preencher valores
  
  # Executar
  deno run dev
  ```
- **Funcionalidades**:
  - Agente conversacional com ferramentas MCP
  - Suporte apenas para respostas tipo texto
  - Debug mode disponível
  - Mensagens salvas em `messages.json`

---

## 📚 Tutoriais e Exemplos

### 🦙 ollama_tutoriais/
**Scripts Python para usar Ollama localmente**

- **Finalidade**: Exemplos práticos de uso do Ollama
- **Como usar**:
  ```bash
  # Verificar se Ollama está rodando
  systemctl status ollama.service
  
  # Executar scripts
  python ollama_basico.py
  python ollama_basico_chat.py
  python ollama_basico_chat_stream.py
  python ollama_chatgpt.py
  ```
- **Scripts disponíveis**:
  - `ollama_basico.py`: Pergunta simples
  - `ollama_basico_chat.py`: Chat básico
  - `ollama_basico_chat_stream.py`: Chat com streaming
  - `ollama_chatgpt.py`: Interface tipo ChatGPT

---

## 🎯 Projetos Especializados

### 🎮 minecraft-mcp/
**Cliente Minecraft customizado (AMCP)**

- **Finalidade**: Cliente Minecraft 1.8.9 modificado para PvP
- **Como compilar**:
  ```bash
  git clone [repositório]
  gradlew build
  ```
- **Funcionalidades**:
  - Base Minecraft 1.8.9
  - OptiFine integrado
  - Discord Rich Presence
  - Suporte Twitch
  - Otimizações Windows
- **Requisitos**: Java 8, Gradle, Windows
- **⚠️ Aviso**: Apenas para fins educacionais

### 🚀 ideias-para-mcps/
**Plataforma MCP Cloud - Projeto Conceito**

- **Finalidade**: Plataforma tipo "Vercel para MCPs" com monetização
- **Status**: Em desenvolvimento/planejamento
- **Funcionalidades planejadas**:
  - Deploy automático de servidores MCP
  - Marketplace de MCPs
  - Sistema de pagamentos DogePay
  - Infraestrutura cloud-native
  - Planos: Gratuito (R$0), Premium (R$99/mês), Enterprise (R$2.000/mês)
- **Documentação**: Múltiplos arquivos `.md` com especificações detalhadas

---

## 📖 Recursos e Documentação

### 📝 system-prompts-and-models-of-ai-tools/
**System prompts de ferramentas IA famosas**

- **Finalidade**: Coleção de prompts internos de ferramentas como v0, Cursor, Manus
- **Conteúdo**: Mais de 5.500 linhas de prompts oficiais
- **Pastas disponíveis**:
  - v0 Folder
  - Manus Folder
  - Same.dev Folder
  - Lovable Folder
  - Devin AI Folder
  - Cursor Folder
- **Como usar**: Estude os prompts para entender como essas ferramentas funcionam

### 🎨 v0-system-prompts-and-models/
**Prompts específicos do v0 da Vercel**

- **Finalidade**: Documentação completa dos prompts do v0
- **Conteúdo**: 2.200+ linhas de prompts oficiais
- **Arquivos**:
  - `v0.txt`: Prompts completos do sistema
  - `v0_model.txt`: Detalhes técnicos dos modelos
- **Modelos usados**:
  - Padrão: GPT-4o
  - Raciocínio avançado: DeepSeek
  - Busca futura: Sonar (Perplexity)

---

## 🚀 Como Começar

### Para Desenvolvimento com IA:
1. **codex/** - Comece aqui para programação assistida por IA
2. **ollama_tutoriais/** - Para modelos locais
3. **openai-mcp-client/** - Para integrar ferramentas MCP

### Para Experimentar Modelos:
1. **Llama3_2_Conversational.ipynb** - Fine-tuning no Colab
2. **modelo-llama-cpp-python/** - Execução local
3. **modelo-bitnet-b1.58-2B-4T/** - Modelo ultra-eficiente

### Para Automação:
1. **github-mcp-server/** - Automação GitHub
2. **cryo-mcp/** - Dados blockchain
3. **chatbot-deepseek-interface/** - Interface conversacional

### Para Aprender:
1. **system-prompts-and-models-of-ai-tools/** - Como grandes ferramentas funcionam
2. **v0-system-prompts-and-models/** - Engenharia de prompt avançada

---

## ⚠️ Avisos Importantes

- **Custos**: Ferramentas que usam APIs podem gerar custos significativos
- **Segurança**: Sempre configure adequadamente chaves de API
- **Hardware**: Alguns modelos requerem GPU ou muita RAM
- **Licenças**: Verifique as licenças antes de uso comercial
- **Atualizações**: Alguns projetos são experimentais e podem mudar

---

## 🆘 Suporte

Para dúvidas específicas sobre cada projeto:
1. Consulte o README.md de cada pasta
2. Verifique a documentação oficial dos projetos
3. Procure issues nos repositórios originais
4. Para projetos locais, verifique os logs de erro

**Última atualização**: Dezembro 2025