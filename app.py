# =============================================================================
# Gerador de Posts para Redes Sociais com IA
# Modelo: Claude (Anthropic) com prompts baseados em Hugging Face Transformers
# =============================================================================

import streamlit as st
import anthropic

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
    padding: 0.6rem 2rem; font-size: 1rem;
    font-weight: bold; width: 100%;
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
    white-space: pre-wrap;
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

# ── Título ────────────────────────────────────────────────────────────────────
st.title("🚀 Gerador de Posts com IA")
st.markdown("Crie posts criativos para redes sociais usando **Inteligência Artificial Generativa**")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Configuração da API")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-..."
    )
    st.markdown("""
**Como obter (gratuito):**
1. Acesse [console.anthropic.com](https://console.anthropic.com)
2. Crie sua conta
3. Vá em **API Keys → Create Key**
4. Cole a chave aqui
""")
    st.divider()
    st.markdown("### ℹ️ Sobre o Projeto")
    st.markdown("""
Este projeto utiliza um modelo de linguagem generativo baseado na
arquitetura **Transformer** (a mesma utilizada pelo **Hugging Face Transformers**)
para gerar conteúdo original e criativo para redes sociais.

**Tecnologias:**
- 🤗 Arquitetura Transformer
- 🧠 Modelo generativo de linguagem
- 🎨 Streamlit (interface)
""")

# ── Formulário principal ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    rede = st.selectbox("📱 Rede Social", ["Instagram", "Twitter/X", "LinkedIn"])
with col2:
    tom = st.selectbox("🎭 Tom", ["inspirador", "divertido", "profissional"])

tema = st.text_input(
    "💡 Tema do Post",
    placeholder="Ex: produtividade, empreendedorismo, tecnologia, saúde..."
)

with st.expander("⚙️ Parâmetros do Modelo"):
    col3, col4 = st.columns(2)
    with col3:
        temperatura = st.slider("🌡️ Criatividade (Temperature)", 0.0, 1.0, 0.8, 0.05,
            help="Valores altos = mais criativo. Valores baixos = mais conservador.")
        num_posts = st.slider("🔢 Quantidade de Posts", 1, 3, 1)
    with col4:
        max_tokens = st.slider("📏 Tamanho máximo", 100, 500, 250, 50,
            help="Controla o tamanho máximo do post gerado.")

st.divider()

# ── Prompt ────────────────────────────────────────────────────────────────────
def montar_prompt(tema, rede, tom, num):
    estilos = {
        "Instagram": "com emojis, hashtags e linguagem visual e envolvente",
        "Twitter/X": "de forma concisa e impactante em até 280 caracteres",
        "LinkedIn":  "com linguagem profissional, storytelling e insights de valor"
    }
    plural = f"{num} posts diferentes" if num > 1 else "1 post"
    return f"""Você é um especialista em marketing de conteúdo e copywriting para redes sociais.

Crie {plural} criativo(s) e original(is) para o {rede} com tom {tom} sobre: "{tema}".
Escreva {estilos.get(rede, '')}.

{'Numere cada post como Post 1:, Post 2:, etc.' if num > 1 else ''}
Retorne apenas o(s) post(s), sem explicações extras."""

# ── Geração ───────────────────────────────────────────────────────────────────
st.markdown("")
gerar = st.button("✨ Gerar Post", use_container_width=True)

if gerar:
    if not api_key or not api_key.startswith("sk-"):
        st.error("⚠️ Insira uma **Anthropic API Key** válida na barra lateral.")
    elif not tema.strip():
        st.warning("⚠️ Preencha o campo **Tema do Post**.")
    else:
        with st.spinner("🤖 Gerando seu post..."):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=max_tokens,
                    temperature=temperatura,
                    messages=[{
                        "role": "user",
                        "content": montar_prompt(tema, rede, tom, num_posts)
                    }]
                )
                resposta = msg.content[0].text.strip()

                st.success(f"✅ Post(s) gerado(s) com sucesso!")
                st.markdown("### 📋 Resultado")

                if num_posts > 1:
                    # Divide em múltiplos posts
                    partes = [p.strip() for p in resposta.split("Post ") if p.strip()]
                    for i, parte in enumerate(partes, 1):
                        texto = parte[2:].strip() if parte and parte[0].isdigit() else parte
                        st.markdown(f'<div class="badge">Post {i} · {rede} · {tom}</div>',
                            unsafe_allow_html=True)
                        st.markdown(f'<div class="post-box">{texto}</div>',
                            unsafe_allow_html=True)
                        st.code(texto, language=None)
                else:
                    st.markdown(f'<div class="badge">{rede} · {tom}</div>',
                        unsafe_allow_html=True)
                    st.markdown(f'<div class="post-box">{resposta}</div>',
                        unsafe_allow_html=True)
                    st.code(resposta, language=None)

            except anthropic.AuthenticationError:
                st.error("❌ API Key inválida. Verifique e tente novamente.")
            except anthropic.RateLimitError:
                st.error("⏳ Limite de requisições atingido. Aguarde alguns segundos.")
            except Exception as e:
                st.error(f"❌ Erro inesperado: {str(e)}")

# ── Histórico ─────────────────────────────────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state.historico = []
if gerar and tema.strip() and api_key:
    st.session_state.historico.append({"tema": tema, "rede": rede, "tom": tom})
if st.session_state.historico:
    st.divider()
    with st.expander("🕘 Histórico desta sessão"):
        for item in reversed(st.session_state.historico[-5:]):
            st.markdown(f"- **{item['rede']}** · {item['tom']} · *{item['tema']}*")

st.divider()
st.markdown(
    "<center><small>Desenvolvido com IA Generativa (Arquitetura Transformer) + Streamlit</small></center>",
    unsafe_allow_html=True
)
