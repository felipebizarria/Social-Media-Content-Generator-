# 🚀 Gerador de Posts para Redes Sociais

Aplicação de IA generativa que cria posts criativos para redes sociais usando o modelo GPT-2 em português, desenvolvida com Hugging Face Transformers e Streamlit.

## 🎯 Sobre o Projeto

Este projeto utiliza o modelo pré-treinado `pierreguillou/gpt2-small-portuguese` para gerar posts originais e criativos para diferentes redes sociais (Instagram, Twitter/X, LinkedIn) a partir de um tema fornecido pelo usuário.

## 🛠️ Tecnologias Utilizadas

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — biblioteca principal para carregar e usar o modelo de IA
- [PyTorch](https://pytorch.org/) — framework de deep learning usado pela biblioteca Transformers
- [Streamlit](https://streamlit.io/) — framework para criar a interface web interativa

## 📁 Estrutura do Projeto

```
social-post-generator/
│
├── app.py                  # Aplicação principal Streamlit (interface)
├── requirements.txt        # Bibliotecas necessárias
├── README.md               # Este arquivo
│
├── model/
│   └── generator.py        # Lógica de geração de texto com Hugging Face
│
├── utils/
│   └── prompts.py          # Templates de prompts para cada rede social
│
└── data/
    └── exemplos.txt        # Exemplos de posts para referência
```

## ▶️ Como Executar Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/social-post-generator.git

# 2. Entre na pasta
cd social-post-generator

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a aplicação
streamlit run app.py
```

## 🌐 Deploy

Aplicação disponível em: [link do Streamlit Cloud]

## 📚 Como Usar

1. Acesse a aplicação
2. Digite o tema do post (ex: "lançamento de produto", "dica de produtividade")
3. Escolha a rede social desejada
4. Ajuste os parâmetros de criatividade
5. Clique em **Gerar Post** e veja o resultado!
