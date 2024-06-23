# -*- coding: utf-8 -*-

# 패키지 가져오기
import streamlit as st
import pandas as pd
import pickle

# matplotlib 백엔드를 설정합니다.
import matplotlib
matplotlib.use('Agg')
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
    st.subheader("Advertising 'sales' 예측기")
    st.markdown(":blue[**AI Wonder**] 제공")

    # 사용자 입력
    digital = st.number_input("digital", value=0.3)
    TV = st.number_input("TV", value=155.62)
    radio = st.number_input("radio", value=7.89)
    newspaper = st.number_input("newspaper", value=6.13)

    st.markdown("")

    # 입력값으로 데이터 만들기
    point = pd.DataFrame([{
        'digital': digital,
        'TV': TV,
        'radio': radio,
        'newspaper': newspaper,
    }])

    # 컨텍스트 로드
    state, model = load_context('advertising')

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
