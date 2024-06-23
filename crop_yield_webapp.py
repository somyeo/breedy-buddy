# -*- coding: utf-8 -*-
pip show matplotlib
# 패키지 가져오기
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import ai_wonder as wonder

# 로더 함수
@st.cache_resource
def load_context(dataname):
    state = wonder.load_state(f"{dataname}_state.pkl")
    model = wonder.input_piped_model(state)
    return state, model

# 드라이버 함수
if __name__ == "__main__":
    # 스트림릿 인터페이스
    st.subheader(f"Crop Yield 'avg_temp' 예측기")
    st.markdown(":blue[**AI Wonder**] 제공")

    # 사용자 입력
    Area = st.text_input("Area", value="Ukraine")
    Item = st.selectbox("Item", ['Rice, paddy', 'Wheat', 'Sorghum', 'Potatoes', 'Sweet potatoes', 'Maize', 'Soybeans', 'Cassava', 'Plantains and others', 'Yams'], index=2)
    hghayield = st.number_input("hg_ha_yield", value=13730)
    averagerainfallmmperyear = st.number_input("average_rain_fall_mm_per_year", value=2280.0)
    pesticidestonnes = st.number_input("pesticides_tonnes", value=4581.73)

    st.markdown("")

    # 입력값으로 데이터 만들기
    point = pd.DataFrame([{
        'Area': Area,
        'Item': Item,
        'hg_ha_yield': hghayield,
        'average_rain_fall_mm_per_year': averagerainfallmmperyear,
        'pesticides_tonnes': pesticidestonnes,
    }])

    # 컨텍스트 로드
    state, model = load_context('crop_yield')

    # 예측 및 설명
    if st.button('예측'):
        st.markdown("")

        with st.spinner("추론 중..."):
            prediction = str(model.predict(point)[0])
            st.success(f"**{state.target}**의 예측값은 **{round(float(prediction), 3)}** 입니다.")
            st.markdown("")

        with st.spinner("설명 생성 중..."):
            st.info("피처 중요도")
            importances = pd.DataFrame(wonder.local_explanations(state, point), columns=["피처", "값", "중요도"])
            st.dataframe(importances.round(2))
