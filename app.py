# app.py
import streamlit as st
from model.generator import carregar_modelo, gerar_post
from utils.prompts import get_prompt

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Gerador de Posts com IA",
    page_icon="🚀",
    layout="centered"
)

# ── Estilo visual ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        width: 100%;
    }
    .post-box {
        background: #1a1a2e;
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .badge {
        display: inline-block;
        background: #667eea22;
        color: #667eea;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Cabeçalho ───────────────────────────────────────────────────────────────
st.title("🚀 Gerador de Posts com IA")
st.markdown("Crie posts criativos para redes sociais usando **Inteligência Artificial**")
st.divider()

# ── Carregamento do modelo (cache — só carrega uma vez) ─────────────────────
@st.cache_resource(show_spinner="⏳ Carregando modelo de IA... (pode levar 1-2 min na primeira vez)")
def get_model():
    return carregar_modelo()

gerador = get_model()

# ── Formulário principal ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    rede_social = st.selectbox(
        "📱 Rede Social",
        options=["Instagram", "Twitter/X", "LinkedIn"],
        help="Cada rede tem um estilo de escrita diferente"
    )

with col2:
    tom = st.selectbox(
        "🎭 Tom do Post",
        options=["inspirador", "divertido", "profissional"],
        help="Define o estilo e a linguagem do post gerado"
    )

tema = st.text_input(
    "💡 Tema do Post",
    placeholder="Ex: dica de produtividade, lançamento de produto, motivação...",
    help="Escreva o assunto principal do post"
)

st.divider()

# ── Parâmetros avançados ─────────────────────────────────────────────────────
with st.expander("⚙️ Parâmetros Avançados"):
    st.markdown("Ajuste esses valores para controlar o comportamento da IA:")
    col3, col4 = st.columns(2)

    with col3:
        temperatura = st.slider("🌡️ Criatividade (Temperatura)", 0.5, 1.4, 0.9, 0.05,
            help="Valores altos = mais criativo. Valores baixos = mais conservador.")
        num_posts = st.slider("🔢 Quantidade de Posts", 1, 3, 1,
            help="Gerar mais posts permite comparar e escolher o melhor")

    with col4:
        max_tokens = st.slider("📏 Tamanho do Texto", 40, 200, 100, 10,
            help="Controla o comprimento máximo do post gerado")
        top_p = st.slider("🎲 Diversidade (Top-P)", 0.5, 1.0, 0.92, 0.02,
            help="Controla a variedade de palavras usadas")

# ── Botão de geração ─────────────────────────────────────────────────────────
st.markdown("")
gerar = st.button("✨ Gerar Post", use_container_width=True)

# ── Lógica de geração ────────────────────────────────────────────────────────
if gerar:
    if not tema.strip():
        st.warning("⚠️ Por favor, preencha o campo **Tema do Post** antes de gerar.")
    else:
        with st.spinner("🤖 A IA está criando seu post..."):
            prompt = get_prompt(tema, rede_social, tom)
            posts = gerar_post(
                gerador=gerador,
                prompt=prompt,
                max_tokens=max_tokens,
                temperatura=temperatura,
                top_p=top_p,
                num_posts=num_posts,
            )

        st.success(f"✅ {len(posts)} post(s) gerado(s) com sucesso!")
        st.markdown("### 📋 Resultado(s)")

        for i, post in enumerate(posts, 1):
            st.markdown(f'<div class="badge">Post {i} • {rede_social} • {tom}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="post-box">{post}</div>', unsafe_allow_html=True)
            st.code(post, language=None)

# ── Histórico na sessão ──────────────────────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state.historico = []

if gerar and tema.strip():
    st.session_state.historico.append({"tema": tema, "rede": rede_social, "tom": tom})

if st.session_state.historico:
    st.divider()
    with st.expander("🕘 Histórico desta sessão"):
        for item in reversed(st.session_state.historico[-5:]):
            st.markdown(f"- **{item['rede']}** | {item['tom']} | tema: *{item['tema']}*")

# ── Rodapé ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Desenvolvido com 🤗 Hugging Face Transformers + Streamlit</small></center>",
    unsafe_allow_html=True
)
