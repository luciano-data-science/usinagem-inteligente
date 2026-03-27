import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÕES DA PÁGINA ---
st.set_page_config(page_title="Precision AI v1.0", layout="wide")

# --- SISTEMA DE ACESSO RESTRITO ---
def check_password():
    """Retorna True se o utilizador introduziu a senha correta."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # Interface de Login
    st.markdown("### 🔒 Acesso Restrito - Industrial Data Intelligence")
    password = st.text_input("Introduza a senha de acesso para análise:", type="password")
    
    # Busca a senha nas Secrets do Streamlit (veremos abaixo)
    # Se quiser testar localmente antes, pode trocar st.secrets["password"] por "12345"
    if st.button("Acessar"):
        if password == st.secrets["password"]:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("⚠️ Senha incorreta. Acesso negado.")
    return False

# Executa a verificação
if not check_password():
    st.stop()  # Para a execução aqui se a senha não estiver correta

# --- O RESTO DO SEU CÓDIGO (st.title, sidebar, etc.) COMEÇA AQUI ---

# --- MOTOR DE MACHINE LEARNING PREDITIVO ---
def run_predictive_ml(df, target_var, cols_to_exclude):
    df_ml = df.select_dtypes(include=[np.number]).copy()
    to_drop = [c for c in df_ml.columns if any(ex in c for ex in cols_to_exclude) or "Unnamed" in c]
    df_ml = df_ml.drop(columns=[c for c in to_drop if c in df_ml.columns and c != target_var])
    df_ml = df_ml.dropna(subset=[target_var]).dropna(axis=1, how='all')
    
    if len(df_ml) < 10 or df_ml[target_var].std() == 0:
        return None, None, 0
    
    X = df_ml.drop(columns=[target_var])
    y = df_ml[target_var]
    
    if X.empty: return None, None, 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, importances, score

# --- LÓGICA ESTATÍSTICA (PRECISÃO 0.001) ---
def calculate_cpk(data, target_col, lse, lsi):
    series = pd.to_numeric(data[target_col], errors='coerce').dropna()
    mean, std = float(series.mean()), float(series.std())
    
    if std <= 0: 
        return {"cpk": 0.000, "scrap_rate": 0.0, "mean": round(mean, 3), "std": 0, "series": series, "total": len(series), "ucl": round(mean, 3), "lcl": round(mean, 3)}
    
    ucl, lcl = mean + (3 * std), mean - (3 * std)
    cpk = min((lse - mean) / (3 * std), (mean - lsi) / (3 * std))
    out_of_spec = len(series[(series > lse) | (series < lsi)])
    scrap_rate = (out_of_spec / len(series)) * 100
    
    return {
        "mean": round(mean, 3), "std": round(std, 3), 
        "ucl": round(ucl, 3), "lcl": round(lcl, 3),
        "cpk": round(cpk, 3), "scrap_rate": round(scrap_rate, 2), 
        "total": len(series), "series": series
    }

# --- INTERFACE ---
st.title("🎯 Precision AI: Inteligência de Usinagem")
st.markdown("""
    ### Análise Avançada de Dados Industriais e Controle Estatístico
    *Rastreabilidade IATF 16949 & Predição de Causa Raiz com Machine Learning*
""")
st.divider()

with st.sidebar:
    st.header("📂 Ingestão de Dados")
    uploaded_file = st.file_uploader("Upload: CSV, Excel ou TXT", type=["csv", "xlsx", "xls", "txt"])
    st.divider()
    st.subheader("📝 Rastreabilidade")
    nome_inspetor = st.text_input("Responsável Técnico", value="Engenharia de Qualidade")
    num_op = st.text_input("Ordem de Produção (OP)", placeholder="Ex: OP-2024-001")
    num_lote = st.text_input("Número do Lote", placeholder="Ex: LT-9982")
    st.divider()
    custo_peca = st.number_input("Custo Unitário de Refugo (R$)", value=10.0, step=0.50)

if uploaded_file:
    try:
        # Registro de rastreabilidade
        file_details = {"Filename": uploaded_file.name, "Timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

        # LÓGICA DE DECODIFICAÇÃO REFORÇADA (SOLUÇÃO PARA O ERRO UTF-16)
        if not uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_raw = None
            # Tenta codificações comuns em equipamentos industriais
            for enc in ['utf-8-sig', 'utf-16', 'utf-16-le', 'latin1', 'cp1252']:
                try:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, sep=None, engine='python', encoding=enc, on_bad_lines='skip')
                    if not df_raw.empty and df_raw.columns.size > 1:
                        break
                except Exception:
                    continue
            
            if df_raw is None:
                st.error("Falha na decodificação. O arquivo pode estar corrompido ou em um formato não suportado.")
                st.stop()
        else:
            df_raw = pd.read_excel(uploaded_file)

        # Heurística de Limites
        specs = {"LSI": {}, "LSE": {}}
        data_start_row = 0
        for i in range(min(len(df_raw), 15)):
            tag = str(df_raw.iloc[i, 0]).strip().upper()
            if any(t in tag for t in ["LSS", "LSE", "SUPERIOR"]): specs["LSE"] = df_raw.iloc[i].to_dict(); data_start_row = i+1
            elif any(t in tag for t in ["LSI", "INFERIOR"]): specs["LSI"] = df_raw.iloc[i].to_dict(); data_start_row = i+1
        
        df_clean = df_raw.iloc[data_start_row:].reset_index(drop=True).copy()
        for col in df_clean.columns:
            if "DATA" in col.upper(): df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')
            else: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        with st.expander("🔍 Janela de Dados Brutos"):
            st.dataframe(df_clean, use_container_width=True)

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("⚙️ Seletor de Variáveis")
            c1, c2, c3 = st.columns(3)
            target_var = c1.selectbox("Variável Alvo", numeric_cols)
            v_lsi = float(pd.to_numeric(specs["LSI"].get(target_var), errors='coerce') or df_clean[target_var].min())
            v_lse = float(pd.to_numeric(specs["LSE"].get(target_var), errors='coerce') or df_clean[target_var].max())
            lsi = c2.number_input("LSI", value=v_lsi, format="%.3f")
            lse = c3.number_input("LSE", value=v_lse, format="%.3f")

            if st.button("🚀 Iniciar Análise Avançada"):
                cols_excluir = ['Ora', 'ID Pezzo']
                stats = calculate_cpk(df_clean, target_var, lse, lsi)
                model, importances, ml_score = run_predictive_ml(df_clean, target_var, cols_excluir)
                prejuizo = (stats['scrap_rate']/100) * stats['total'] * custo_peca
                
                # DASHBOARD CONSOLIDADO
                st.divider()
                st.subheader("📈 Indicadores de Performance")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Amostra Total", f"{stats['total']}")
                m2.metric("Capabilidade (Cpk)", f"{stats['cpk']:.3f}")
                m3.metric("Scrap (%)", f"{stats['scrap_rate']}%")
                m4.metric("Perda (R$)", f"R$ {prejuizo:,.2f}")
                m5.metric("Confiabilidade IA", f"{ml_score*100:.1f}%")

                g1, g2 = st.columns(2)
                with g1:
                    fig_h, ax_h = plt.subplots(figsize=(8, 5))
                    sns.histplot(stats['series'], kde=True, ax=ax_h, color='#2E86C1')
                    ax_h.axvline(lsi, color='red', ls='--'); ax_h.axvline(lse, color='red', ls='--')
                    st.pyplot(fig_h)
                with g2:
                    fig_c, ax_c = plt.subplots(figsize=(8, 5))
                    ax_c.plot(stats['series'].values, marker='o', color='#2E86C1', markersize=3, lw=0.5)
                    ax_c.axhline(stats['mean'], color='green', label=f"Média: {stats['mean']:.3f}")
                    ax_c.axhline(stats['ucl'], color='red', ls='--', label=f"LSC: {stats['ucl']:.3f}")
                    ax_c.axhline(stats['lcl'], color='red', ls='--', label=f"LIC: {stats['lcl']:.3f}")
                    ax_c.legend(loc='upper right', fontsize='small')
                    st.pyplot(fig_c)

                # MATRIZ E IA
                st.divider()
                st.subheader("📈 Matriz de Correlação")
                df_corr = df_clean.select_dtypes(include=[np.number]).copy()
                cols_remover = [c for c in df_corr.columns if any(ex in c for ex in cols_excluir) or "Unnamed" in c]
                df_corr = df_corr.drop(columns=cols_remover)
                fig_corr, ax_corr = plt.subplots(figsize=(12, 7))
                sns.heatmap(df_corr.corr(), annot=True, cmap='RdYlGn', fmt=".2f", ax=ax_corr)
                st.pyplot(fig_corr)

                st.divider()
                st.subheader("🔮 Causa Raiz Preditiva (IA)")
                if importances is not None and not importances.empty:
                    fig_ml, ax_ml = plt.subplots(figsize=(10, 5))
                    importances.head(10).plot(kind='barh', ax=ax_ml, color='#28B463')
                    st.pyplot(fig_ml)

                # --- PDF v4.7 ---
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", 'B', 16); pdf.cell(0, 10, "Relatorio de Inteligência de Dados Industriais", ln=True, align='C')
                
                # SEÇÃO RASTREABILIDADE
                pdf.ln(5); pdf.set_font("Helvetica", 'B', 12); pdf.set_fill_color(230, 230, 230)
                pdf.cell(0, 10, "0. Rastreabilidade (IATF 16949)", ln=True, fill=True)
                pdf.set_font("Helvetica", '', 10)
                pdf.cell(0, 7, f"Responsavel: {nome_inspetor}", ln=True)
                pdf.cell(0, 7, f"Ordem de Producao (OP): {num_op if num_op else 'N/A'}", ln=True)
                pdf.cell(0, 7, f"Numero do Lote: {num_lote if num_lote else 'N/A'}", ln=True)
                pdf.cell(0, 7, f"Arquivo: {file_details['Filename']} | Analise em: {file_details['Timestamp']}", ln=True)
                
                # SEÇÃO PERFORMANCE
                pdf.ln(5); pdf.set_font("Helvetica", 'B', 12); pdf.cell(0, 10, "1. Performance e Impacto", ln=True, fill=True)
                pdf.set_font("Helvetica", '', 10)
                impacto_data = [
                    ["Variavel Alvo:", str(target_var)], ["Amostra:", f"{stats['total']} pcs"],
                    ["Cpk:", f"{stats['cpk']:.3f}"], ["Scrap:", f"{stats['scrap_rate']}%"],
                    ["Perda Financeira:", f"R$ {prejuizo:,.2f}"], ["Media:", f"{stats['mean']:.3f}"],
                    ["LIC:", f"{stats['lcl']:.3f}"], ["LSC:", f"{stats['ucl']:.3f}"]
                ]
                for item in impacto_data:
                    pdf.set_font("Helvetica", 'B', 10); pdf.cell(50, 7, item[0])
                    pdf.set_font("Helvetica", '', 10); pdf.cell(0, 7, item[1], ln=True)

                def save_to_pdf(fig, title):
                    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
                    pdf.add_page(); pdf.set_font("Helvetica", 'B', 12); pdf.cell(0, 10, title, ln=True)
                    pdf.image(buf, x=15, w=175); return buf

                save_to_pdf(fig_h, "2. Grafico de Capabilidade")
                save_to_pdf(fig_c, "3. Carta de Controle & Análise de Tendência")
                save_to_pdf(fig_corr, "4. Matriz de Correlacao Global")
                if importances is not None: save_to_pdf(fig_ml, "5. Ranking de Influência gerado por IA")
                
                # --- RODAPÉ DO RELATÓRIO PDF ---
                pdf.add_page() # Opcional: ou apenas adicione ao final da última página
                pdf.ln(10)
                pdf.set_font("Helvetica", 'I', 8)
                pdf.set_text_color(128, 128, 128) # Cor cinza para um visual profissional
                footer_text = "Powered by Luciano Martins Teixeira - Industrial Data Intelligence Consultant - LOGI SERVICE DO BRASIL - 2026"
                pdf.cell(0, 10, footer_text, ln=True, align='C')
                st.download_button("📥 Baixar Relatório Técnico", bytes(pdf.output()), f"Relatorio_Auditoria_{target_var}.pdf", "application/pdf")

    except Exception as e: st.error(f"Erro Crítico: {e}")

# --- RODAPÉ DA PÁGINA ---
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.8em;'>
        Powered by Luciano Martins Teixeira - Industrial Data Intelligence Consultant - 
        <b>LOGI SERVICE DO BRASIL</b> - 2026
    </div>
    """, 
    unsafe_allow_html=True
)