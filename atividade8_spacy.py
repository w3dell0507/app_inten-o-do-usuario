import streamlit as st
import spacy
import pandas as pd

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Classificador de Intenções NLP",
    page_icon="🤖",
    layout="centered"
)

# Carregamento do modelo spaCy com tratamento de erro
@st.cache_resource
def carregar_spacy():
    try:
        return spacy.load("pt_core_news_sm")
    except OSError:
        # Se não encontrar localmente, faz o download automático do modelo
        from spacy.cli import download
        download("pt_core_news_sm")
        return spacy.load("pt_core_news_sm")

nlp = carregar_spacy()

# Dicionário de palavras-chave/lemmas mapeados para cada intenção
INTENCOES = {
    "Bloquear Conta / Cartão": {"bloquear", "bloqueio", "perdi", "roubo", "roubar", "furtar", "furto", "perda", "segurança"},
    "Solicitar 2ª Via": {"via", "segunda", "fatura", "boleto", "segunda-via", "reemitir", "codigo", "linha"},
    "Comprar / Contratar": {"comprar", "adquirir", "contratar", "preço", "valor", "plano", "assinar", "comprar-novo"},
    "Cancelar Serviço": {"cancelar", "desistir", "encerrar", "reembolso", "devolver", "cancelamento"},
    "Suporte Técnico": {"erro", "problema", "defeito", "bug", "ajuda", "suporte", "senha", "acesso", "funciona"}
}

def identificar_intencao(texto):
    doc = nlp(texto)
    scores = {intencao: 0 for intencao in INTENCOES}
    
    # Processa cada token e compara com o lema correspondente
    detalhes_tokens = []
    for token in doc:
        if not token.is_punct and not token.is_space:
            lema = token.lemma_.lower()
            intencao_encontrada = None
            
            for intencao, palavras in INTENCOES.items():
                if lema in palavras:
                    scores[intencao] += 1
                    intencao_encontrada = intencao
            
            detalhes_tokens.append({
                "Token": token.text,
                "Lema (Raiz)": lema,
                "Classe Gramatical": token.pos_,
                "Intenção Mapeada": intencao_encontrada if intencao_encontrada else "-"
            })
            
    # Determina a intenção com maior pontuação
    intencao_final = max(scores, key=scores.get)
    if scores[intencao_final] == 0:
        intencao_final = "Não Identificada / Atendimento Geral"
        
    return intencao_final, detalhes_tokens, doc

# Interface Visual
st.title("🤖 Classificador de Intenções de Clientes")
st.markdown("Digite a mensagem do cliente para identificar a ação necessária (Bloqueio, 2ª Via, Suporte, etc.).")

# Entrada do utilizador
mensagem = st.text_area("Mensagem do Cliente:", placeholder="Ex: Perdi meu cartão e preciso bloquear ou solicitar a segunda via da fatura...")

if st.button("Analisar Intenção", type="primary"):
    if mensagem.strip():
        intencao, tokens_df, doc = identificar_intencao(mensagem)
        
        st.divider()
        st.subheader("🎯 Resultado da Análise")
        
        # Exibição da Intenção em destaque
        if "Bloquear" in intencao:
            st.error(f"🚨 **AÇÃO REQUERIDA:** {intencao}")
        elif "2ª Via" in intencao:
            st.warning(f"📄 **AÇÃO REQUERIDA:** {intencao}")
        elif "Cancelar" in intencao:
            st.warning(f"⚠️ **AÇÃO REQUERIDA:** {intencao}")
        elif "Comprar" in intencao:
            st.success(f"🛒 **AÇÃO REQUERIDA:** {intencao}")
        else:
            st.info(f"ℹ️ **AÇÃO REQUERIDA:** {intencao}")
            
        # Exibição dos Tokens e Lemas processados pelo spaCy
        st.subheader("🔍 Tabela de Lemas e Tokens (spaCy)")
        st.dataframe(pd.DataFrame(tokens_df), use_container_width=True)
        
    else:
        st.warning("Por favor, digite uma mensagem antes de analisar.")
