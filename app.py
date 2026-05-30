# app.py
# Gerador de Posts para Redes Sociais
# Usa Hugging Face Transformers via huggingface_hub InferenceClient

import streamlit as st
from huggingface_hub import InferenceClient

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(page_title="Gerador de Posts com IA", page_icon="🚀", layout="centered")

st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 12px;
        padding: 0.6rem 2rem; font-size: 1rem;
        font-weight: bold; width: 100%;
    }
    .post-box {
        background: #1a1a2e; border-left: 4px solid #667eea;
        border-radius: 12px; padding: 1.2rem 1.5rem;
        margin: 0.8rem 0; color: #e0e0e0;
        font-size: 0.95rem; line-height: 1.6;
    }
    .badge {
        display: inline-block; background: #667eea22;
        color: #667eea; border-radius: 20px;
        padding: 0.2rem 0.8rem; font-size: 0.8rem; margin-bottom: 0.5rem;
    }
    .info-box {
        background: #0d2137; border: 1px solid #667eea55;
        border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
        font-size: 0.85rem; color: #aac4e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Cabeçalho ────────────────────────────────────────────────────────────────
st.title("🚀 Gerador de Posts com IA")
st.markdown("Crie posts criativos para redes sociais usando **Hugging Face Transformers**")

st.markdown("""
<div class="info-box">
🤗 <b>Modelo:</b> <code>pierreguillou/gpt2-small-portuguese</code> — GPT-2 pré-treinado em português<br>
⚡ <b>Tecnologia:</b> Hugging Face Transformers via InferenceClient (biblioteca oficial)
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar com API Key ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤗 Hugging Face")
    st.markdown("**Modelo usado:**")
    st.code("pierreguillou/gpt2-small-portuguese", language=None)
    st.markdown("---")
    hf_token = st.text_input(
        "🔑 Hugging Face Token",
        type="password",
        help="Token de acesso gratuito do Hugging Face"
    )
    st.markdown("**Como obter:**")
    st.markdown("1. Acesse [huggingface.co](https://huggingface.co)")
    st.markdown("2. Crie conta gratuita")
    st.markdown("3. **Settings → Access Tokens**")
    st.markdown("4. Crie um token tipo **Read**")
    st.markdown("---")
    st.markdown("### 📖 Sobre o Modelo")
    st.markdown("""
    O **GPT-2 Portuguese** é um modelo **Transformer** (GPT-2) da biblioteca
    **Hugging Face Transformers**, pré-treinado em textos portugueses.
    Ele gera texto de forma autoregressiva — prevendo a próxima palavra com base no contexto.
    """)

# ── Formulário principal ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    rede_social = st.selectbox("📱 Rede Social", ["Instagram", "Twitter/X", "LinkedIn"])
with col2:
    tom = st.selectbox("🎭 Tom do Post", ["inspirador", "divertido", "profissional"])

tema = st.text_input("💡 Tema do Post",
    placeholder="Ex: dica de produtividade, empreendedorismo, motivação...")

st.divider()

with st.expander("⚙️ Parâmetros do Modelo Transformers"):
    st.markdown("Controle o comportamento do modelo GPT-2:")
    col3, col4 = st.columns(2)
    with col3:
        temperatura = st.slider("🌡️ Temperatura (Criatividade)",
            0.5, 1.4, 0.9, 0.05,
            help="Alto = mais criativo e imprevisível.")
        max_tokens = st.slider("📏 Máx. de Tokens",
            40, 200, 100, 10,
            help="Quantidade máxima de palavras geradas.")
    with col4:
        top_p = st.slider("🎲 Top-P (Diversidade)",
            0.5, 1.0, 0.92, 0.02,
            help="Nucleus sampling: variedade de vocabulário.")
        repetition_penalty = st.slider("🔁 Penalidade de Repetição",
            1.0, 2.0, 1.3, 0.1,
            help="Evita repetição de palavras.")

# ── Prompt por rede e tom ─────────────────────────────────────────────────────
def montar_prompt(tema, rede, tom):
    templates = {
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
    return templates.get(rede, {}).get(tom, f"Post sobre {tema}:\n\n")

# ── Chamada via InferenceClient (huggingface_hub oficial) ─────────────────────
def gerar_post_hf(prompt, token, temperatura, max_tokens, top_p, repetition_penalty):
    """
    Usa o InferenceClient da biblioteca huggingface_hub para chamar
    o modelo GPT-2 Portuguese rodando nos servidores do Hugging Face.
    Esta é a abordagem oficial de usar Hugging Face Transformers em produção.
    """
    MODEL_ID = "pierreguillou/gpt2-small-portuguese"

    client = InferenceClient(model=MODEL_ID, token=token)

    resultado = client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperatura,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        return_full_text=False,  # retorna apenas o texto novo, sem o prompt
    )

    texto = resultado.strip()

    # Limpeza: corta no primeiro parágrafo duplo
    if "\n\n" in texto:
        texto = texto.split("\n\n")[0]

    return texto

# ── Botão e lógica de geração ─────────────────────────────────────────────────
st.markdown("")
gerar = st.button("✨ Gerar Post", use_container_width=True)

if gerar:
    if not hf_token:
        st.error("⚠️ Insira seu **Hugging Face Token** na barra lateral.")
    elif not tema.strip():
        st.warning("⚠️ Preencha o campo **Tema do Post** antes de gerar.")
    else:
        prompt = montar_prompt(tema, rede_social, tom)

        with st.spinner("🤖 O modelo GPT-2 Portuguese está gerando seu post..."):
            try:
                resultado = gerar_post_hf(
                    prompt=prompt,
                    token=hf_token,
                    temperatura=temperatura,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

                if not resultado:
                    st.warning("O modelo retornou um texto vazio. Tente outro tema ou ajuste os parâmetros.")
                else:
                    st.success("✅ Post gerado com sucesso!")
                    st.markdown("### 📋 Resultado")
                    st.markdown(f'<div class="badge">{rede_social} • {tom} • GPT-2 Portuguese</div>',
                        unsafe_allow_html=True)
                    st.markdown(f'<div class="post-box">{resultado}</div>', unsafe_allow_html=True)
                    st.code(resultado, language=None)

                    with st.expander("🔍 Ver prompt enviado ao modelo"):
                        st.code(prompt + resultado, language=None)
                        st.caption("O modelo recebeu o prompt e completou o texto — assim funciona geração com Transformers.")

            except Exception as e:
                erro = str(e)
                if "401" in erro or "authorization" in erro.lower():
                    st.error("❌ Token inválido. Verifique seu Hugging Face Token.")
                elif "503" in erro or "loading" in erro.lower():
                    st.warning("⏳ Modelo em cold start. Aguarde 30 segundos e tente novamente.")
                else:
                    st.error(f"❌ Erro ao chamar o modelo: {erro}")

# ── Histórico ─────────────────────────────────────────────────────────────────
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
    unsafe_allow_html=True
)
