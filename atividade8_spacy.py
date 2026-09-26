import streamlit as st
import spacy
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Classificador de Intenções NLP",
    page_icon="🤖",
    layout="centered"
)

# Inicialização do histórico na sessão do Streamlit
if "historico" not in st.session_state:
    st.session_state.historico = []

# Carregamento do modelo spaCy
@st.cache_resource
def carregar_spacy():
    try:
        return spacy.load("pt_core_news_sm")
    except OSError:
        from spacy.cli import download
        download("pt_core_news_sm")
        return spacy.load("pt_core_news_sm")

nlp = carregar_spacy()

# Mapeamento de intenções e palavras-chave
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
            
    intencao_final = max(scores, key=scores.get)
    if scores[intencao_final] == 0:
        intencao_final = "Não Identificada / Atendimento Geral"
        
    return intencao_final, detalhes_tokens

# Interface Visual
st.title("🤖 Classificador de Intenções com Histórico")
st.markdown("Digite a mensagem do cliente para identificar a intenção e acompanhar o histórico de análises.")

# Entrada do utilizador
mensagem = st.text_area("Mensagem do Cliente:", placeholder="Ex: Preciso bloquear o meu cartão e pedir a 2 via da fatura...")

col1, col2 = st.columns([3, 1])
with col1:
    btn_analisar = st.button("Analisar Intenção", type="primary", use_container_width=True)
with col2:
    btn_limpar = st.button("Limpar Histórico", use_container_width=True)

if btn_limpar:
    st.session_state.historico = []
    st.rerun()

if btn_analisar:
    if mensagem.strip():
        intencao, tokens_df = identificar_intencao(mensagem)
        
        # Guarda no histórico (a entrada mais recente fica no início)
        st.session_state.historico.insert(0, {
            "mensagem": mensagem,
            "intencao": intencao,
            "tokens": tokens_df
        })
    else:
        st.warning("Por favor, digite uma mensagem antes de analisar.")

# Exibição dos resultados e histórico
if st.session_state.historico:
    st.divider()
    st.subheader("📜 Histórico de Atendimentos")
    
    for i, item in enumerate(st.session_state.historico):
        # Destaque visual de acordo com a intenção
        emoji = "🚨" if "Bloquear" in item["intencao"] else "📄" if "2ª Via" in item["intencao"] else "⚠️" if "Cancelar" in item["intencao"] else "🛒" if "Comprar" in item["intencao"] else "ℹ️"
        
        with st.expander(f"{emoji} **Consulta #{len(st.session_state.historico) - i}:** {item['intencao']}", expanded=(i == 0)):
            st.write(f"**Mensagem analisada:** *\"{item['mensagem']}\"*")
            st.markdown(f"**Intenção Detetada:** `{item['intencao']}`")
            st.markdown("**Análise de Tokens e Lemas (spaCy):**")
            st.dataframe(pd.DataFrame(item["tokens"]), use_container_width=True)
