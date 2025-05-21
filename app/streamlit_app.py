import streamlit as st
import pandas as pd
import joblib

st.title("QSAR Drug Repurposing App")

uploaded_file = st.file_uploader("Загрузите df_repurposing.xlsx", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Пример загруженных данных:", df.head())

    # Удалим мета- и нечисловые колонки
    meta_cols = ['ChEMBL_ID', 'ROMol', 'Name', 'Indication', 'canonical_smiles']
    columns_to_keep = [col for col in df.columns if col not in meta_cols and pd.api.types.is_numeric_dtype(df[col])]
    X = df[columns_to_keep]

    # Загрузка моделей
    recall_pipeline = joblib.load("models/recall_screening.pkl")
    precision_pipeline = joblib.load("models/precision_screening.pkl")

    # Этап 1: Предсказание активности
    recall_preds = recall_pipeline.predict(X)
    df['KRAS_G12D_activity'] = recall_preds
    st.write("🧪 Этап 1: Предсказание активности (recall screening)")
    st.write("Количество активных молекул:", (df['KRAS_G12D_activity'] == 1).sum())
    st.dataframe(df[df['KRAS_G12D_activity'] == 1].head())

    # Этап 2: Предсказание специфичности среди активных
    active_df = df[df['KRAS_G12D_activity'] == 1]
    if not active_df.empty:
        X_active = X.loc[active_df.index]
        kras_preds = precision_pipeline.predict(X_active)

        # Запись предсказаний
        df.loc[active_df.index, 'KRAS_specificity'] = kras_preds
        st.success("✅ Этап 2 завершён: KRAS специфичность определена")
        st.write("🧬 Результаты специфичности (precision screening)")
        st.dataframe(df.loc[active_df.index, ['KRAS_G12D_activity', 'KRAS_specificity']].head())

        # Сохранение
        import io
        from pandas import ExcelWriter
        buffer = io.BytesIO()
        with ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
            writer.save()
        st.download_button(
            label="⬇️ Скачать результаты (Excel)",
            data=buffer,
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("❗ Нет активных молекул — пропускаем этап 2.")
