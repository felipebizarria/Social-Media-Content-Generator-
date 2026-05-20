# utils/prompts.py
# Este arquivo contém os templates de prompts para cada rede social.
# Um "prompt" é o texto que enviamos para a IA como ponto de partida.
# Quanto melhor o prompt, melhor o resultado gerado.

def get_prompt(tema: str, rede_social: str, tom: str) -> str:
    """
    Monta o prompt ideal para cada combinação de rede social e tom.
    
    Parâmetros:
        tema: assunto do post (ex: "dica de produtividade")
        rede_social: plataforma alvo (Instagram, Twitter, LinkedIn)
        tom: estilo do texto (inspirador, divertido, profissional)
    
    Retorna:
        Uma string com o prompt completo para enviar ao modelo
    """

    templates = {
        "Instagram": {
            "inspirador": f"Post inspirador para o Instagram sobre {tema}:\n\n✨",
            "divertido":  f"Post divertido e descontraído para o Instagram sobre {tema}:\n\n😄",
            "profissional": f"Post profissional para o Instagram sobre {tema}:\n\n📌",
        },
        "Twitter/X": {
            "inspirador": f"Tweet inspirador sobre {tema} (máximo 280 caracteres):\n\n🧵",
            "divertido":  f"Tweet engraçado e criativo sobre {tema}:\n\n😂",
            "profissional": f"Tweet profissional e direto sobre {tema}:\n\n💼",
        },
        "LinkedIn": {
            "inspirador": f"Post inspirador para o LinkedIn sobre {tema}, voltado para profissionais:\n\n🚀",
            "divertido":  f"Post leve e humanizado para o LinkedIn sobre {tema}:\n\n😊",
            "profissional": f"Post formal e analítico para o LinkedIn sobre {tema}:\n\n📊",
        },
    }

    # Retorna o template ou um genérico se não encontrar
    return templates.get(rede_social, {}).get(
        tom,
        f"Crie um post criativo sobre {tema} para {rede_social}:\n\n"
    )
