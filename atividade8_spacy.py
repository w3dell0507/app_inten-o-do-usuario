import streamlit as st
import spacy

# Carregamento do modelo em português
@st.cache_resource
def carregar_spacy():
    return spacy.load("pt_core_news_sm")

nlp = carregar_spacy()

st.title("🤖 Detecção de Intenção do Usuário com spaCy")
st.write("Classificação de intenção (*Comprar*, *Cancelar*, *Suporte*) baseada na lematização das palavras.")

# Mapeamento de lemas (raízes) para cada intenção
INTENCOES = {
    "comprar": {"comprar", "adquirir", "contratar", "pedido", "preço", "valor", "assinar", "compra"},
    "cancelar": {"cancelar", "desistir", "encerrar", "cancelamento", "reembolso", "devolver", "estorno"},
    "suporte": {"erro", "problema", "defeito", "bug", "ajuda", "suporte", "travar", "socorro", "senha", "acesso"}
}

# Entrada do texto do usuário
mensagem = st.text_input(
    "Digite sua mensagem para o chatbot:",
    placeholder="Ex: Quero cancelar o meu plano e pedir reembolso."
)

if st.button("Identificar Intenção", type="primary"):
    if not mensagem.strip():
        st.warning("Por favor, digite uma mensagem.")
    else:
        # Processamento do texto pelo spaCy
        doc = nlp(mensagem)
        
        # Extração dos lemas em minúsculas (ignorando pontuações e espaços)
        lemas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
        
        # Contagem de correspondências por intenção
        scores = {intencao: 0 for intencao in INTENCOES}
        palavras_detectadas = {intencao: [] for intencao in INTENCOES}
        
        for token in doc:
            if not token.is_punct and not token.is_space:
                lema = token.lemma_.lower()
                for intencao, palavras_chave in INTENCOES.items():
                    if lema in palavras_chave:
                        scores[intencao] += 1
                        palavras_detectadas[intencao].append(token.text)
        
        # Determinar a intenção principal
        intencao_definida = max(scores, key=scores.get)
        max_score = scores[intencao_definida]
        
        st.subheader("Resultado do Reconhecimento:")
        
        if max_score == 0:
            st.info("❓ **Intenção: DESCONHECIDA** (Nenhuma palavra-chave identificada)")
        else:
            if intencao_definida == "comprar":
                st.success(f"🛒 **Intenção Detectada: COMPRAR**")
            elif intencao_definida == "cancelar":
                st.error(f"🚫 **Intenção Detectada: CANCELAR**")
            elif intencao_definida == "suporte":
                st.warning(f"🛠️ **Intenção Detectada: SUPORTE**")
                
            st.write(f"Termos que ativaram a intenção: **{', '.join(palavras_detectadas[intencao_definida])}**")

        # Exibição dos lemas extraídos pelo spaCy
        with st.expander("🔍 Detalhes do Processamento NLP (spaCy)"):
            st.write("**Tokens e Lemas extraídos:**")
            tabela_tokens = [{"Palavra Original": token.text, "Lema (Raiz)": token.lemma_.lower()} for token in doc if not token.is_punct and not token.is_space]
            st.dataframe(tabela_tokens, use_container_width=True)