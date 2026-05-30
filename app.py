# app.py — Gerador de Posts para Redes Sociais
# Hugging Face Transformers via InferenceClient (huggingface_hub 0.32)

import streamlit as st
from huggingface_hub import InferenceClient

MODEL_ID = "pierreguillou/gpt2-small-portuguese"

st.set_page_config(page_title="Gerador de Posts com IA", page_icon="🚀", layout="centered")

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
        font-size: 0.95rem; line-height: 1.6;
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
</style>
""", unsafe_allow_html=True)

st.title("🚀 Gerador de Posts com IA")
st.markdown("Crie posts criativos para redes sociais usando **Hugging Face Transformers**")
st.markdown(f"""
<div class="info-box">
🤗 <b>Modelo:</b> <code>{MODEL_ID}</code> — GPT-2 pré-treinado em português<br>
⚡ <b>Tecnologia:</b> Hugging Face Transformers · InferenceClient
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤗 Hugging Face")
    st.code(MODEL_ID, language=None)
    st.markdown("---")
    hf_token = st.text_input("🔑 Hugging Face Token", type="password")
    st.markdown("""
**Como obter (gratuito):**
1. Acesse [huggingface.co](https://huggingface.co)
2. Crie uma conta
3. **Settings → Access Tokens**
4. Clique em **New token** → tipo **Read**
5. Cole o token aqui
""")
    st.markdown("---")
    st.markdown("### 📖 Sobre o Modelo")
    st.markdown("""
**GPT-2 Portuguese** é um modelo da família **Transformer** da biblioteca
**Hugging Face Transformers**, pré-treinado em corpus de textos em português.
Gera texto prevendo a próxima palavra com base no contexto — geração autoregressiva.
""")

# ── Formulário ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    rede_social = st.selectbox("📱 Rede Social", ["Instagram", "Twitter/X", "LinkedIn"])
with col2:
    tom = st.selectbox("🎭 Tom do Post", ["inspirador", "divertido", "profissional"])

tema = st.text_input("💡 Tema do Post",
    placeholder="Ex: dica de produtividade, empreendedorismo, motivação...")
st.divider()

with st.expander("⚙️ Parâmetros do Modelo Transformers"):
    col3, col4 = st.columns(2)
    with col3:
        temperatura  = st.slider("🌡️ Temperatura",    0.5, 1.4, 0.9, 0.05)
        max_tokens   = st.slider("📏 Máx. Tokens",     40, 200, 100, 10)
    with col4:
        top_p        = st.slider("🎲 Top-P",           0.5, 1.0, 0.92, 0.02)
        rep_penalty  = st.slider("🔁 Penalidade Rep.", 1.0, 2.0, 1.3, 0.1)

# ── Prompt ─────────────────────────────────────────────────────────────────────
def montar_prompt(tema, rede, tom):
    t = {
        "Instagram": {
            "inspirador":   f"Post inspirador para Instagram sobre {tema}:\n\n✨",
            "divertido":    f"Post divertido para Instagram sobre {tema}:\n\n😄",
            "profissional": f"Post profissional para Instagram sobre {tema}:\n\n📌",
        },
        "Twitter/X": {
            "inspirador":   f"Tweet inspirador sobre {tema}:\n\n🧵",
            "divertido":    f"Tweet engraçado sobre {tema}:\n\n😂",
            "profissional": f"Tweet profissional sobre {tema}:\n\n💼",
        },
        "LinkedIn": {
            "inspirador":   f"Post inspirador para LinkedIn sobre {tema}:\n\n🚀",
            "divertido":    f"Post leve para LinkedIn sobre {tema}:\n\n😊",
            "profissional": f"Post formal para LinkedIn sobre {tema}:\n\n📊",
        },
    }
    return t.get(rede, {}).get(tom, f"Post sobre {tema}:\n\n")

# ── Geração ─────────────────────────────────────────────────────────────────────
def gerar_post(prompt, token, temperatura, max_tokens, top_p, rep_penalty):
    """
    Usa InferenceClient da huggingface_hub 0.32 que roteia pelo endpoint
    router.huggingface.co — acessível no Streamlit Cloud.
    """
    client = InferenceClient(token=token)

    resultado = client.text_generation(
        prompt,
        model=MODEL_ID,
        max_new_tokens=max_tokens,
        temperature=float(temperatura),
        top_p=float(top_p),
        repetition_penalty=float(rep_penalty),
        do_sample=True,
        return_full_text=False,
    )

    texto = resultado.strip()
    if "\n\n" in texto:
        texto = texto.split("\n\n")[0]
    return texto

# ── Botão ───────────────────────────────────────────────────────────────────────
st.markdown("")
gerar = st.button("✨ Gerar Post", use_container_width=True)

if gerar:
    if not hf_token:
        st.error("⚠️ Insira seu **Hugging Face Token** na barra lateral.")
    elif not tema.strip():
        st.warning("⚠️ Preencha o campo **Tema do Post**.")
    else:
        prompt = montar_prompt(tema, rede_social, tom)
        with st.spinner("🤖 O modelo GPT-2 Portuguese está gerando seu post..."):
            try:
                resultado = gerar_post(
                    prompt, hf_token, temperatura, max_tokens, top_p, rep_penalty
                )
                if not resultado:
                    st.warning("O modelo retornou texto vazio. Tente outro tema.")
                else:
                    st.success("✅ Post gerado com sucesso!")
                    st.markdown("### 📋 Resultado")
                    st.markdown(
                        f'<div class="badge">{rede_social} • {tom} • GPT-2 Portuguese</div>',
                        unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="post-box">{resultado}</div>',
                        unsafe_allow_html=True)
                    st.code(resultado, language=None)
                    with st.expander("🔍 Prompt enviado ao modelo"):
                        st.code(prompt + resultado, language=None)
                        st.caption("O modelo recebeu o prompt e completou o texto — geração autoregressiva com Transformers.")

            except Exception as e:
                err = str(e)
                if "401" in err or "unauthorized" in err.lower() or "authorization" in err.lower():
                    st.error("❌ Token inválido. Verifique seu Hugging Face Token.")
                elif "503" in err or "loading" in err.lower():
                    st.warning("⏳ Modelo em cold start. Aguarde 30 segundos e tente novamente.")
                elif "resolve" in err.lower() or "connection" in err.lower() or "network" in err.lower():
                    st.error("❌ Erro de rede. Tente novamente em alguns segundos.")
                else:
                    st.error(f"❌ Erro ao gerar post: {err}")

# ── Histórico ───────────────────────────────────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state.historico = []
if gerar and tema.strip() and hf_token:
    st.session_state.historico.append({"tema": tema, "rede": rede_social, "tom": tom})
if st.session_state.historico:
    st.divider()
    with st.expander("🕘 Histórico desta sessão"):
        for item in reversed(st.session_state.historico[-5:]):
            st.markdown(f"- **{item['rede']}** | {item['tom']} | tema: *{item['tema']}*")

st.divider()
st.markdown(
    "<center><small>Desenvolvido com 🤗 Hugging Face Transformers + Streamlit</small></center>",
    unsafe_allow_html=True)
