import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def predict_sum(text, model):

    if model == "BART-Small":
        model_name = "prillarosaria/bart-small-indosum"
    elif model == "T5-Small":
        model_name = "prillarosaria/t5-small-indosum"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    input = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input.input_ids, num_beams=3, max_length=250, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    if "xa0" in summary:
        summary = summary.replace("xa0", "")
    return summary

def main():
    summary = ""
    button_style = "background-color: #4CAF50; color: white; width: 200px; height: 50px; font-size: 16px; border-radius: 5px;"
    st.title("Peringkasan Teks Otomatis")

    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        st.subheader("Masukkan teks")

        with st.form(key="form1"):
            model = st.selectbox("Pilih Model:", ["BART-Small", "T5-Small"])
            text = st.text_area("Teks", height=350)
            st.markdown("""
                <style>
                    div.stButton > button:first-child {
                        background-color: rgb(204, 49, 49);
                        color:white;
                        font-size:20px;
                        height:2em;
                        width:23em;
                        border-radius:10px;
                    }
                </style>
            """, unsafe_allow_html=True)
            button = st.form_submit_button(label="Ringkas")

        if button:
            if text:
                summary = predict_sum(text, model)
            else:
                st.warning("Masukkan teks terlebih dahulu")

    with col2:
        st.subheader("Hasil Ringkasan")
        st.write(summary)
        

if __name__ == "__main__":
    main()