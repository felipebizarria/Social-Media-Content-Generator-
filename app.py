# app.py
# Gerador de Posts para Redes Sociais
# Usa Hugging Face Transformers via Inference API
# Modelo: pierreguillou/gpt2-small-portuguese (GPT-2 pré-treinado em português)

import streamlit as st
import requests

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
🤗 <b>Modelo:</b> <code>pierreguillou/gpt2-small-portuguese</code> — GPT-2 pré-treinado em textos portugueses<br>
⚡ <b>Tecnologia:</b> Hugging Face Transformers via Inference API (sem instalação local)
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
        help="Token de acesso do Hugging Face (gratuito)"
    )
    st.markdown("**Como obter:**")
    st.markdown("1. Acesse [huggingface.co](https://huggingface.co)")
    st.markdown("2. Crie conta gratuita")
    st.markdown("3. Vá em **Settings → Access Tokens**")
    st.markdown("4. Crie um token **Read**")
    st.markdown("---")
    st.markdown("### 📖 Sobre o Modelo")
    st.markdown("""
    O **GPT-2 Portuguese** é um modelo da família **GPT-2** (Generative Pre-trained Transformer 2)
    da biblioteca **Hugging Face Transformers**, ajustado com textos em português brasileiro.
    """)

# ── Formulário principal ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    rede_social = st.selectbox("📱 Rede Social",
        ["Instagram", "Twitter/X", "LinkedIn"])
with col2:
    tom = st.selectbox("🎭 Tom do Post",
        ["inspirador", "divertido", "profissional"])

tema = st.text_input("💡 Tema do Post",
    placeholder="Ex: dica de produtividade, empreendedorismo, motivação...")

st.divider()

with st.expander("⚙️ Parâmetros do Modelo Transformers"):
    st.markdown("Controle o comportamento do modelo GPT-2:")
    col3, col4 = st.columns(2)
    with col3:
        temperatura = st.slider(
            "🌡️ Temperatura (Criatividade)",
            min_value=0.5, max_value=1.4, value=0.9, step=0.05,
            help="Controla a aleatoriedade da geração. Alto = mais criativo."
        )
        max_tokens = st.slider(
            "📏 Máx. de Tokens",
            min_value=40, max_value=200, value=100, step=10,
            help="Quantidade máxima de palavras geradas pelo modelo."
        )
    with col4:
        top_p = st.slider(
            "🎲 Top-P (Diversidade)",
            min_value=0.5, max_value=1.0, value=0.92, step=0.02,
            help="Nucleus sampling: controla variedade do vocabulário."
        )
        repetition_penalty = st.slider(
            "🔁 Penalidade de Repetição",
            min_value=1.0, max_value=2.0, value=1.3, step=0.1,
            help="Evita que o modelo repita as mesmas palavras."
        )

# ── Prompts por rede e tom ────────────────────────────────────────────────────
def montar_prompt(tema, rede, tom):
    templates = {
        "Instagram": {
            "inspirador":    f"Post inspirador para Instagram sobre {tema}:\n\n✨",
            "divertido":     f"Post divertido para Instagram sobre {tema}:\n\n😄",
            "profissional":  f"Post profissional para Instagram sobre {tema}:\n\n📌",
        },
        "Twitter/X": {
            "inspirador":    f"Tweet inspirador sobre {tema}:\n\n🧵",
            "divertido":     f"Tweet engraçado sobre {tema}:\n\n😂",
            "profissional":  f"Tweet profissional sobre {tema}:\n\n💼",
        },
        "LinkedIn": {
            "inspirador":    f"Post inspirador para LinkedIn sobre {tema}:\n\n🚀",
            "divertido":     f"Post leve para LinkedIn sobre {tema}:\n\n😊",
            "profissional":  f"Post formal para LinkedIn sobre {tema}:\n\n📊",
        },
    }
    return templates.get(rede, {}).get(tom, f"Post sobre {tema}:\n\n")

# ── Chamada à Hugging Face Inference API ─────────────────────────────────────
def gerar_post_hf(prompt, token, temperatura, max_tokens, top_p, repetition_penalty):
    """
    Chama a Hugging Face Inference API para gerar texto com o modelo GPT-2 Portuguese.
    Esta é a forma de usar Hugging Face Transformers sem instalar torch localmente.
    O modelo roda nos servidores do Hugging Face.
    """
    MODEL_ID = "pierreguillou/gpt2-small-portuguese"
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":      max_tokens,
            "temperature":         temperatura,
            "top_p":               top_p,
            "repetition_penalty":  repetition_penalty,
            "do_sample":           True,
            "return_full_text":    False,   # retorna só o texto NOVO (sem o prompt)
        },
        "options": {
            "wait_for_model": True   # aguarda se o modelo estiver em cold start
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code == 200:
        resultado = response.json()
        if isinstance(resultado, list) and len(resultado) > 0:
            texto = resultado[0].get("generated_text", "").strip()
            # Limpeza: corta em parágrafo duplo
            if "\n\n" in texto:
                texto = texto.split("\n\n")[0]
            return texto
        return "Não foi possível gerar o texto."
    elif response.status_code == 503:
        return "⏳ Modelo em inicialização (cold start). Aguarde 20 segundos e tente novamente."
    elif response.status_code == 401:
        return "❌ Token inválido. Verifique seu Hugging Face Token."
    else:
        return f"❌ Erro {response.status_code}: {response.text}"

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
            resultado = gerar_post_hf(
                prompt=prompt,
                token=hf_token,
                temperatura=temperatura,
                max_tokens=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        if resultado.startswith("❌") or resultado.startswith("⏳"):
            st.warning(resultado)
        else:
            st.success("✅ Post gerado com sucesso!")
            st.markdown("### 📋 Resultado")
            st.markdown(f'<div class="badge">{rede_social} • {tom} • GPT-2 Portuguese</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="post-box">{resultado}</div>', unsafe_allow_html=True)
            st.code(resultado, language=None)

            # Exibe o prompt usado (transparência do modelo)
            with st.expander("🔍 Ver prompt enviado ao modelo"):
                st.code(prompt + resultado, language=None)
                st.caption("O modelo recebeu o prompt acima e gerou o texto em continuação — isso é como funciona a geração de texto com Transformers.")

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
