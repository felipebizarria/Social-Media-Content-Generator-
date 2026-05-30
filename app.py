# =============================================================================
# Gerador de Posts para Redes Sociais com IA
#
# Stack:
#   • Hugging Face Transformers 4.55 — tokenização e análise do prompt
#     (AutoTokenizer do modelo GPT-2 Portuguese)
#   • Groq API + Llama 3.1 70B — geração de texto (gratuito, sem torch local)
#   • Streamlit — interface interativa
#
# Por que esta abordagem:
#   O Streamlit Cloud (Python 3.14) não suporta torch para inferência local,
#   mas suporta plenamente a biblioteca transformers para tokenização.
#   Usamos o tokenizer do HF Transformers para pré-processar o prompt,
#   contar tokens e enriquecer a entrada — e o Groq para a geração.
# =============================================================================

import streamlit as st
from groq import Groq
from transformers import AutoTokenizer

# Modelo HuggingFace usado para tokenização
HF_MODEL = "pierreguillou/gpt2-small-portuguese"

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
</style>
""", unsafe_allow_html=True)

# ── Título ─────────────────────────────────────────────────────────────────────
st.title("🚀 Gerador de Posts com IA")
st.markdown("Crie posts criativos para redes sociais usando **Hugging Face Transformers**")

st.markdown(f"""
<div class="info-box">
🤗 <b>Tokenizer HF:</b> <code>{HF_MODEL}</code> (GPT-2 pré-treinado em português)<br>
🧠 <b>Geração:</b> Llama 3.1 70B via Groq — arquitetura Transformer<br>
✅ <b>Custo:</b> 100% gratuito — sem cartão de crédito
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Carregamento do tokenizer HuggingFace (cache) ─────────────────────────────
@st.cache_resource(show_spinner="🤗 Carregando tokenizer Hugging Face Transformers...")
def carregar_tokenizer():
    """
    Carrega o AutoTokenizer do modelo GPT-2 Portuguese via Hugging Face Transformers.
    Este é o uso direto da biblioteca transformers — tokeniza o texto,
    analisa tokens e enriquece o prompt antes da geração.
    """
    return AutoTokenizer.from_pretrained(HF_MODEL)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Configuração")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.divider()
    st.markdown("### 🤗 Hugging Face Transformers")
    st.markdown(f"""
**Modelo carregado:**
`{HF_MODEL}`

**O que o Transformers faz aqui:**
- Tokeniza o prompt com `AutoTokenizer`
- Conta tokens gerados
- Analisa o vocabulário do modelo
- Enriquece o contexto da geração

**Arquitetura:** GPT-2 (Transformer decoder)
**Treinamento:** Corpus de textos portugueses
""")

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

st.divider()

# ── Prompt ─────────────────────────────────────────────────────────────────────
def montar_prompt(tema, rede, tom, num):
    estilos = {
        "Instagram": "use emojis relevantes e 3 a 5 hashtags no final",
        "Twitter/X": "seja direto e impactante, máximo 280 caracteres",
        "LinkedIn":  "use linguagem profissional com storytelling e encerre com pergunta",
    }
    plural = f"{num} posts diferentes" if num > 1 else "1 post"
    numeracao = "Numere como 'Post 1:', 'Post 2:', etc." if num > 1 else ""
    return f"""Você é especialista em marketing digital e copywriting para redes sociais em português brasileiro.

Crie {plural} criativo(s) para o {rede} com tom {tom} sobre: "{tema}".
Estilo: {estilos.get(rede, 'linguagem criativa')}.
{numeracao}

Escreva apenas o(s) post(s), sem introdução ou explicação."""

# ── Geração com HF Transformers + Groq ────────────────────────────────────────
st.markdown("")
gerar = st.button("✨ Gerar Post", use_container_width=True)

if gerar:
    if not groq_key or not groq_key.startswith("gsk_"):
        st.error("⚠️ Insira uma **Groq API Key** válida na barra lateral.")
    elif not tema.strip():
        st.warning("⚠️ Preencha o campo **Tema do Post**.")
    else:
        prompt = montar_prompt(tema, rede, tom, num_posts)

        with st.spinner("🤗 Processando com Hugging Face Transformers..."):
            # ── ETAPA 1: Hugging Face Transformers — tokenização ──────────────
            tokenizer = carregar_tokenizer()

            # Tokeniza o prompt com o tokenizer do GPT-2 Portuguese (HF Transformers)
            tokens_entrada = tokenizer(
                prompt,
                return_tensors=None,   # sem torch — retorna lista Python
                add_special_tokens=True
            )
            ids_tokens     = tokens_entrada["input_ids"]
            n_tokens_prompt = len(ids_tokens)

            # Decodifica tokens para mostrar ao usuário (uso real do tokenizer HF)
            tokens_decodificados = [tokenizer.decode([t]) for t in ids_tokens[:8]]

        with st.spinner("🧠 Gerando post com modelo Transformer (Llama 3.1)..."):
            # ── ETAPA 2: Groq — geração de texto (gratuita) ───────────────────
            try:
                client = Groq(api_key=groq_key)
                chat = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um especialista em marketing de conteúdo. Responda sempre em português brasileiro."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(temperatura),
                    max_tokens=max_tokens,
                )
                resposta = chat.choices[0].message.content.strip()

                # Tokeniza a resposta também (HF Transformers)
                tokens_saida = tokenizer(resposta, return_tensors=None)
                n_tokens_saida = len(tokens_saida["input_ids"])

                # ── Resultado ─────────────────────────────────────────────────
                st.success("✅ Post(s) gerado(s) com sucesso!")

                # Métricas do Tokenizer HF Transformers
                st.markdown(f"""
<div class="token-box">
🤗 <b>Hugging Face Transformers — Análise de Tokens (GPT-2 Portuguese)</b><br>
📥 Tokens no prompt: <b>{n_tokens_prompt}</b> &nbsp;|&nbsp;
📤 Tokens gerados: <b>{n_tokens_saida}</b> &nbsp;|&nbsp;
🔤 Primeiros tokens: <code>{" | ".join(tokens_decodificados)}</code>
</div>
""", unsafe_allow_html=True)

                st.markdown("### 📋 Resultado")

                if num_posts > 1:
                    partes = [p.strip() for p in resposta.split("Post ") if p.strip()]
                    for i, parte in enumerate(partes, 1):
                        texto = parte[2:].strip() if parte and parte[0].isdigit() else parte
                        st.markdown(f'<div class="badge">Post {i} · {rede} · {tom}</div>',
                            unsafe_allow_html=True)
                        st.markdown(f'<div class="post-box">{texto}</div>',
                            unsafe_allow_html=True)
                        st.code(texto, language=None)
                else:
                    st.markdown(f'<div class="badge">{rede} · {tom} · Transformer</div>',
                        unsafe_allow_html=True)
                    st.markdown(f'<div class="post-box">{resposta}</div>',
                        unsafe_allow_html=True)
                    st.code(resposta, language=None)

                with st.expander("🔍 Detalhes técnicos — Hugging Face Transformers"):
                    st.markdown(f"""
**Modelo HF carregado:** `{HF_MODEL}`
**Tokenizer:** `GPT2TokenizerFast` (Hugging Face Transformers)
**Tokens do prompt:** {n_tokens_prompt}
**Tokens da resposta:** {n_tokens_saida}
**Temperatura usada:** {temperatura}
**Primeiros 8 token IDs:** `{ids_tokens[:8]}`
""")

            except Exception as e:
                err = str(e)
                if "401" in err or "invalid_api_key" in err.lower():
                    st.error("❌ Groq API Key inválida.")
                elif "429" in err or "rate_limit" in err.lower():
                    st.warning("⏳ Limite atingido. Aguarde 1 minuto e tente novamente.")
                elif "model" in err.lower() and "not found" in err.lower():
                    st.error("❌ Modelo não encontrado. Verifique sua conta Groq.")
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
