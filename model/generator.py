# model/generator.py
# Este é o coração do projeto.
# Aqui carregamos o modelo do Hugging Face e geramos os textos.

from transformers import pipeline, set_seed
import torch

# Nome do modelo no Hugging Face Hub
# GPT-2 treinado em textos portugueses — leve e roda sem GPU
MODEL_NAME = "pierreguillou/gpt2-small-portuguese"

def carregar_modelo():
    """
    Baixa e carrega o modelo do Hugging Face.
    Na primeira execução faz o download (~500MB).
    Nas próximas execuções usa o cache local (bem mais rápido).
    
    Retorna:
        Um pipeline de geração de texto pronto para usar
    """
    print(f"Carregando modelo: {MODEL_NAME}")
    
    # "pipeline" é a forma mais simples de usar um modelo no Hugging Face
    # task="text-generation" diz que queremos gerar texto
    # device=-1 força uso de CPU (funciona em qualquer máquina)
    gerador = pipeline(
        task="text-generation",
        model=MODEL_NAME,
        device=-1  # -1 = CPU | 0 = GPU (se disponível)
    )
    
    print("Modelo carregado com sucesso!")
    return gerador


def gerar_post(
    gerador,
    prompt: str,
    max_tokens: int = 120,
    temperatura: float = 0.9,
    top_p: float = 0.92,
    num_posts: int = 1,
    seed: int = None
) -> list[str]:
    """
    Gera posts usando o modelo carregado.
    
    Parâmetros:
        gerador:      o pipeline carregado por carregar_modelo()
        prompt:       texto inicial que a IA vai completar
        max_tokens:   tamanho máximo do texto gerado (em tokens/palavras)
        temperatura:  criatividade — valores altos = mais criativo, mais aleatório
                      valores baixos = mais conservador, mais repetitivo
                      range recomendado: 0.5 (sério) a 1.4 (muito criativo)
        top_p:        diversidade de vocabulário (0.0 a 1.0)
                      valores altos = usa palavras mais variadas
        num_posts:    quantos posts diferentes gerar ao mesmo tempo
        seed:         número para reproduzir exatamente o mesmo resultado
                      None = resultado diferente a cada geração
    
    Retorna:
        Lista de strings com os posts gerados (sem o prompt inicial)
    """
    
    # Se seed fornecida, garante reprodutibilidade
    if seed is not None:
        set_seed(seed)
    
    # Chama o modelo para gerar o texto
    resultados = gerador(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperatura,
        top_p=top_p,
        num_return_sequences=num_posts,
        do_sample=True,          # True = geração criativa (amostragem)
        pad_token_id=50256,      # token especial de padding do GPT-2
        repetition_penalty=1.2,  # penaliza repetições no texto gerado
    )
    
    # Extrai apenas o texto NOVO gerado (remove o prompt do início)
    posts = []
    for resultado in resultados:
        texto_completo = resultado["generated_text"]
        texto_novo = texto_completo[len(prompt):].strip()
        
        # Limpeza básica: remove texto após linha em branco dupla
        if "\n\n" in texto_novo:
            texto_novo = texto_novo.split("\n\n")[0]
        
        posts.append(texto_novo)
    
    return posts
