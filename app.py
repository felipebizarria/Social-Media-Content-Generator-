# app.py
# Arquivo principal da aplicação.
# Execute com: streamlit run app.py

import streamlit as st
from model.generator import carregar_modelo, gerar_post
from utils.prompts import get_prompt

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Gerador de Posts com IA",
    page_icon="🚀",
    layout="centered"
)

# ── Estilo visual customizado ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
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

# ── Carregamento do modelo (só faz isso UMA vez graças ao @st.cache_resource) ──
# @st.cache_resource garante que o modelo é carregado uma única vez
# mesmo que o usuário interaja várias vezes com a interface
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

# ── Parâmetros avançados (ficam escondidos por padrão) ──────────────────────
with st.expander("⚙️ Parâmetros Avançados"):
    st.markdown("Ajuste esses valores para controlar o comportamento da IA:")
    
    col3, col4 = st.columns(2)
    
    with col3:
        temperatura = st.slider(
            "🌡️ Criatividade (Temperatura)",
            min_value=0.5,
            max_value=1.4,
            value=0.9,
            step=0.05,
            help="Valores altos = mais criativo e imprevisível. Valores baixos = mais conservador e previsível."
        )
        num_posts = st.slider(
            "🔢 Quantidade de Posts",
            min_value=1,
            max_value=3,
            value=1,
            help="Gerar mais posts permite comparar e escolher o melhor"
        )
    
    with col4:
        max_tokens = st.slider(
            "📏 Tamanho do Texto",
            min_value=40,
            max_value=200,
            value=100,
            step=10,
            help="Controla o comprimento máximo do post gerado"
        )
        top_p = st.slider(
            "🎲 Diversidade (Top-P)",
            min_value=0.5,
            max_value=1.0,
            value=0.92,
            step=0.02,
            help="Controla a variedade de palavras usadas"
        )

# ── Botão de geração ─────────────────────────────────────────────────────────
st.markdown("")
gerar = st.button("✨ Gerar Post", use_container_width=True)

# ── Lógica de geração ────────────────────────────────────────────────────────
if gerar:
    if not tema.strip():
        st.warning("⚠️ Por favor, preencha o campo **Tema do Post** antes de gerar.")
    else:
        with st.spinner("🤖 A IA está criando seu post..."):
            # Monta o prompt com base nas escolhas do usuário
            prompt = get_prompt(tema, rede_social, tom)
            
            # Chama o modelo para gerar os posts
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
            # Botão para copiar o texto
            st.code(post, language=None)

# ── Histórico na sessão ──────────────────────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state.historico = []

if gerar and tema.strip():
    st.session_state.historico.append({
        "tema": tema,
        "rede": rede_social,
        "tom": tom
    })

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
