# =============================================================================
# Gerador de Posts para Redes Sociais com IA
#
# Stack:
#   • Hugging Face Transformers 4.55 — tokenização com AutoTokenizer
#   • Groq API + Llama 3.3 70B — geração de texto (gratuito)
#   • Dataset próprio (data/dataset.json) — few-shot prompting
#   • Streamlit — interface interativa
# =============================================================================

import json
import random
import streamlit as st
from groq import Groq
from transformers import AutoTokenizer

HF_MODEL = "pierreguillou/gpt2-small-portuguese"
DATASET_PATH = "data/dataset.json"

st.set_page_config(
    page_title="Gerador de Posts com IA",
    page_icon="🚀",
    layout="centered"
)

st.markdown("""
<style>
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 12px;
    padding: 0.6rem 2rem; font-size: 1rem; font-weight: bold; width: 100%;
}
.post-box {
    background: #1a1a2e; border-left: 4px solid #667eea; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin: 0.8rem 0; color: #e0e0e0;
    font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap;
}
.badge {
    display: inline-block; background: #667eea22; color: #667eea;
    border-radius: 20px; padding: 0.2rem 0.8rem;
    font-size: 0.8rem; margin-bottom: 0.5rem;
}
.info-box {
    background: #0d2137; border: 1px solid #667eea55; border-radius: 10px;
    padding: 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #aac4e0;
}
.token-box {
    background: #0a1628; border: 1px solid #444; border-radius: 8px;
    padding: 0.7rem 1rem; font-size: 0.8rem; color: #88aacc; margin-top: 0.5rem;
}
.example-box {
    background: #0f1f10; border-left: 3px solid #4caf50; border-radius: 8px;
    padding: 0.8rem 1rem; margin: 0.4rem 0; font-size: 0.82rem;
    color: #c8e6c9; white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ── Título ─────────────────────────────────────────────────────────────────────
st.title("🚀 Gerador de Posts com IA")
st.markdown("Crie posts criativos para redes sociais usando **Hugging Face Transformers**")

st.markdown(f"""
<div class="info-box">
🤗 <b>Tokenizer HF:</b> <code>{HF_MODEL}</code> (GPT-2 pré-treinado em português)<br>
🧠 <b>Geração:</b> Llama 3.3 70B via Groq — arquitetura Transformer<br>
📚 <b>Dataset:</b> Conjunto próprio de posts para few-shot prompting<br>
✅ <b>Custo:</b> 100% gratuito
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Carregamento do tokenizer HuggingFace ─────────────────────────────────────
@st.cache_resource(show_spinner="🤗 Carregando tokenizer Hugging Face Transformers...")
def carregar_tokenizer():
    return AutoTokenizer.from_pretrained(HF_MODEL)

# ── Carregamento do dataset ───────────────────────────────────────────────────
@st.cache_data
def carregar_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def buscar_exemplos(dataset, rede, tom, n=2):
    """
    Busca exemplos do dataset que correspondem à rede social e tom escolhidos.
    Esses exemplos são injetados no prompt como few-shot examples —
    guiando o modelo a gerar no estilo correto.
    """
    # Filtra exemplos que combinam rede + tom
    combinam = [
        e for e in dataset["exemplos"]
        if e["rede"] == rede and e["tom"] == tom
    ]
    # Se não houver exemplos suficientes, pega só pela rede
    if len(combinam) < n:
        combinam = [e for e in dataset["exemplos"] if e["rede"] == rede]
    # Retorna até n exemplos aleatórios
    return random.sample(combinam, min(n, len(combinam)))

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Configuração")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.divider()
    st.markdown("### 🤗 Hugging Face Transformers")
    st.markdown(f"""
**Modelo:** `{HF_MODEL}`

**Uso no projeto:**
- `AutoTokenizer` tokeniza o prompt
- Conta tokens de entrada e saída
- Exibe token IDs do modelo GPT-2

**Arquitetura:** GPT-2 (Transformer decoder)
""")
    st.divider()
    st.markdown("### 📚 Dataset")
    try:
        ds = carregar_dataset()
        st.markdown(f"""
**Arquivo:** `data/dataset.json`
**Total de exemplos:** {ds['total_exemplos']}
**Versão:** {ds['versao']}

Exemplos reais usados para **few-shot prompting** — ensinam o modelo o estilo esperado para cada rede e tom.
""")
    except Exception:
        st.warning("Dataset não carregado.")

# ── Formulário ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    rede = st.selectbox("📱 Rede Social", ["Instagram", "Twitter/X", "LinkedIn"])
with col2:
    tom  = st.selectbox("🎭 Tom", ["inspirador", "divertido", "profissional"])

tema = st.text_input(
    "💡 Tema do Post",
    placeholder="Ex: produtividade, empreendedorismo, tecnologia, saúde..."
)

with st.expander("⚙️ Parâmetros do Modelo Transformer"):
    col3, col4 = st.columns(2)
    with col3:
        temperatura = st.slider("🌡️ Temperatura", 0.0, 1.0, 0.8, 0.05,
            help="Alto = mais criativo. Baixo = mais conservador.")
        num_posts   = st.slider("🔢 Quantidade de Posts", 1, 3, 1)
    with col4:
        max_tokens  = st.slider("📏 Máx. Tokens", 100, 600, 300, 50)
        usar_dataset = st.toggle("📚 Usar exemplos do dataset", value=True,
            help="Injeta exemplos reais no prompt para guiar o estilo da geração.")

st.divider()

# ── Montagem do prompt com few-shot ────────────────────────────────────────────
def montar_prompt(tema, rede, tom, num, exemplos):
    estilos = {
        "Instagram": "use emojis relevantes e 3 a 5 hashtags no final",
        "Twitter/X": "seja direto e impactante, máximo 280 caracteres",
        "LinkedIn":  "use linguagem profissional com storytelling e encerre com pergunta",
    }
    plural     = f"{num} posts diferentes" if num > 1 else "1 post"
    numeracao  = "Numere como 'Post 1:', 'Post 2:', etc." if num > 1 else ""

    # Few-shot: exemplos do dataset injetados no prompt
    bloco_exemplos = ""
    if exemplos:
        bloco_exemplos = "\n\n**Exemplos do dataset para referência de estilo:**\n"
        for i, ex in enumerate(exemplos, 1):
            bloco_exemplos += f"\nExemplo {i} (tema: {ex['tema']}):\n{ex['post']}\n"
        bloco_exemplos += "\n---\nAgora gere um post original sobre o tema solicitado, no mesmo estilo dos exemplos acima.\n"

    return f"""Você é especialista em marketing digital e copywriting para redes sociais em português brasileiro.
{bloco_exemplos}
Crie {plural} criativo(s) para o {rede} com tom {tom} sobre: "{tema}".
Estilo: {estilos.get(rede, 'linguagem criativa')}.
{numeracao}

Escreva apenas o(s) post(s), sem introdução ou explicação."""

# ── Botão e geração ────────────────────────────────────────────────────────────
st.markdown("")
gerar = st.button("✨ Gerar Post", use_container_width=True)

if gerar:
    if not groq_key or not groq_key.startswith("gsk_"):
        st.error("⚠️ Insira uma **Groq API Key** válida na barra lateral.")
    elif not tema.strip():
        st.warning("⚠️ Preencha o campo **Tema do Post**.")
    else:
        # Carrega dataset e busca exemplos
        try:
            ds       = carregar_dataset()
            exemplos = buscar_exemplos(ds, rede, tom) if usar_dataset else []
        except Exception:
            exemplos = []

        prompt = montar_prompt(tema, rede, tom, num_posts, exemplos)

        with st.spinner("🤗 Processando com Hugging Face Transformers..."):
            tokenizer       = carregar_tokenizer()
            tokens_entrada  = tokenizer(prompt, return_tensors=None, add_special_tokens=True)
            ids_tokens      = tokens_entrada["input_ids"]
            n_tokens_prompt = len(ids_tokens)
            tokens_decoded  = [tokenizer.decode([t]) for t in ids_tokens[:8]]

        with st.spinner("🧠 Gerando post com Llama 3.3 (Transformer)..."):
            try:
                client = Groq(api_key=groq_key)
                chat   = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é especialista em marketing de conteúdo. Responda sempre em português brasileiro."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(temperatura),
                    max_tokens=max_tokens,
                )
                resposta       = chat.choices[0].message.content.strip()
                tokens_saida   = tokenizer(resposta, return_tensors=None)
                n_tokens_saida = len(tokens_saida["input_ids"])

                st.success("✅ Post(s) gerado(s) com sucesso!")

                # Métricas do tokenizer HF
                st.markdown(f"""
<div class="token-box">
🤗 <b>Hugging Face Transformers — Análise de Tokens (GPT-2 Portuguese)</b><br>
📥 Tokens no prompt: <b>{n_tokens_prompt}</b> &nbsp;|&nbsp;
📤 Tokens gerados: <b>{n_tokens_saida}</b> &nbsp;|&nbsp;
🔤 Primeiros tokens: <code>{" | ".join(tokens_decoded)}</code>
</div>
""", unsafe_allow_html=True)

                # Exemplos usados do dataset
                if exemplos:
                    with st.expander(f"📚 {len(exemplos)} exemplo(s) do dataset usados no prompt"):
                        for ex in exemplos:
                            st.markdown(
                                f"**{ex['rede']} · {ex['tom']} · tema: {ex['tema']}**")
                            st.markdown(
                                f'<div class="example-box">{ex["post"]}</div>',
                                unsafe_allow_html=True)

                st.markdown("### 📋 Resultado")

                if num_posts > 1:
                    partes = [p.strip() for p in resposta.split("Post ") if p.strip()]
                    for i, parte in enumerate(partes, 1):
                        texto = parte[2:].strip() if parte and parte[0].isdigit() else parte
                        st.markdown(
                            f'<div class="badge">Post {i} · {rede} · {tom}</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="post-box">{texto}</div>',
                            unsafe_allow_html=True)
                        st.code(texto, language=None)
                else:
                    st.markdown(
                        f'<div class="badge">{rede} · {tom} · Llama 3.3</div>',
                        unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="post-box">{resposta}</div>',
                        unsafe_allow_html=True)
                    st.code(resposta, language=None)

                with st.expander("🔍 Detalhes técnicos"):
                    st.markdown(f"""
**Modelo HF:** `{HF_MODEL}`
**Tokenizer:** `GPT2TokenizerFast`
**Tokens prompt:** {n_tokens_prompt}
**Tokens resposta:** {n_tokens_saida}
**Temperatura:** {temperatura}
**Exemplos do dataset usados:** {len(exemplos)}
**Few-shot ativo:** {"Sim" if usar_dataset and exemplos else "Não"}
""")

            except Exception as e:
                err = str(e)
                if "401" in err or "invalid_api_key" in err.lower():
                    st.error("❌ Groq API Key inválida.")
                elif "429" in err or "rate_limit" in err.lower():
                    st.warning("⏳ Limite atingido. Aguarde 1 minuto e tente novamente.")
                elif "decommissioned" in err or "not found" in err.lower():
                    st.error("❌ Modelo não disponível. Contate o suporte.")
                else:
                    st.error(f"❌ Erro: {err}")

# ── Histórico ──────────────────────────────────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state.historico = []
if gerar and tema.strip() and groq_key:
    st.session_state.historico.append({"tema": tema, "rede": rede, "tom": tom})
if st.session_state.historico:
    st.divider()
    with st.expander("🕘 Histórico desta sessão"):
        for item in reversed(st.session_state.historico[-5:]):
            st.markdown(f"- **{item['rede']}** · {item['tom']} · *{item['tema']}*")

st.divider()
st.markdown(
    "<center><small>Desenvolvido com 🤗 Hugging Face Transformers + Groq + Streamlit</small></center>",
    unsafe_allow_html=True
)
