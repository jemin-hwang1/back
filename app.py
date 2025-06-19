import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import re
import random
import uuid
from utils import highlight_excluded_rows_factory
from collections import OrderedDict, defaultdict


# 엑셀 파일 경로
excel_path = "./final_stat_summary.xlsx"  # 실제 경로에 맞게 수정
excluded_ids = {1,5,6}
# 엑셀 시트 로드
excel_data = pd.read_excel(excel_path, sheet_name="Sheet1")

# 딕셔너리 생성: (risk_code, prompt_code) -> {count, mean_score, weighted_mean_score}
final_stat_dict = {}

def dic_return():
    # 각 row의 prompt_code 기준으로 순회
    for _, row in excel_data.iterrows():
        prompt_code = row['prompt_code'].strip()  # 앞뒤 공백 제거

        # RP 파생형을 'RP'로 통합 처리        
        if prompt_code.startswith("pRP"):
            print("✅ 파생 RP 코드 발견 → 변경 전:", prompt_code)
            prompt_code = "pRP"

        # r01 ~ r35 반복
        for i in range(1, 36):
            risk_code = f"r{i:02d}"
            count_key = f"{risk_code}_count"
            sum_base_score = f"{risk_code}_sum_base_score"
            weighted_key = f"{risk_code}_weighted_score"

            if pd.notna(row.get(count_key)) or pd.notna(row.get(sum_base_score)) or pd.notna(row.get(weighted_key)):
                if sum_base_score in row:
                    mean_score = float(row[sum_base_score]) if pd.notna(row[sum_base_score]) else 0.0
                else:
                    print(f"❌ 열 없음: {sum_base_score}")
                    mean_score = 0.0
                final_stat_dict[(risk_code, prompt_code)] = {
                    "count": int(row[count_key]) if pd.notna(row[count_key]) else 0,
                    "sum_base_score": float(row[sum_base_score]) if pd.notna(row[sum_base_score]) else 0.0,
                    "weighted_mean_score": float(row[weighted_key]) if pd.notna(row[weighted_key]) else 0.0,
                }

    # 결과 확인 (예: 상위 5개)
    for k, v in list(final_stat_dict.items())[:5]:
        print(k, v)

st.set_page_config(page_title="LLM Risk Heatmap Dashboard", layout="wide")

st.markdown("""
    <style>
        .main-title {
            font-size:40px;
            font-weight:bold;
            color:#333333;
            text-align:center;
            margin-bottom: 20px;
        }
        .description {
            font-size:18px;
            color:#666666;
            text-align:center;
            margin-bottom: 30px;
        }
        .footer {
            font-size:14px;
            color:#888888;
            text-align:center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }
        .dimmed {
            color: #aaaaaa;
        }
        .dim-cell {
            background-color: #f5f5f5 !important;
            color: #bbbbbb !important;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">🧠 LLM Safety Score Heatmap</div>', unsafe_allow_html=True)
st.markdown('<div class="description"> AssureAI의 Safety Score를 시각화합니다.<br>이 대시보드는 모델의 위험 분석 및 사전배포 평가 결과를 탐색하는 데 활용됩니다.</div>', unsafe_allow_html=True)

# --------------------------
# 위험 카테고리 및 데이터 구성
# --------------------------
risk_types = OrderedDict({
    "r01": "1. Supporting Malicious Organized Groups",
    "r02": "2. Celebrating Suffering",
    "r03": "3. Violent Acts",
    "r04": "4. Depicting Violence",
    "r05": "5. Weapon Usage & Development",
    "r06": "6. Military and Warfare",
    "r07": "7. Harassment",
    "r08": "8. Hate Speech",
    "r09": "9. Offensive Language",
    "r10": "10. Perpetuating Harmful Beliefs",
    "r11": "11. Adult Content",
    "r12": "12. Erotic Content",
    "r13": "13. Non-Consensual Nudity",
    "r14": "14. Monetized Sexual Content",
    "r15": "15. Endangerment, Harm, or Abuse of Children",
    "r16": "16. Child Sexual Abuse",
    "r17": "17. Suicidal and Non-suicidal Self-injury",
    "r18": "18. Political Persuasion",
    "r19": "19. Influencing Politics",
    "r20": "20. Deterring Democratic Participation",
    "r21": "21. Fraud",
    "r22": "22. Mis/disinformation",
    "r23": "23. Sowing Division",
    "r24": "24. Misrepresentation",
    "r25": "25. Types of Defamation",
    "r26": "26. Discriminatory Activities",
    "r27": "27. Unauthorized Privacy Violations",
    "r28": "28. Illegal/Regulated Substances",
    "r29": "29. Illegal Services/Exploitation",
    "r30": "30. Other Unlawful/Criminal Activities",
    "r31": "31. Increased inequality and decline in employment quality",
    "r32": "32. Economic and cultural devaluation of human effort",
    "r33": "33. Competitive dynamics",
    "r34": "34. Overreliance and unsafe use",
    "r35": "35. Loss of human agency and autonomy"
})
    
prompt_types =  OrderedDict({
    "pMC": "Multiple-Choice",
    "pQO": "Q Only",
    "pMS": "Multi-Session",
    "pRP": "Role-Playing",   # RPemo, RPedu, RPfun 등은 모두 이걸로 통합
    "pCT": "Chain of Thought",
    "pEP": "Expert Prompting",
    "pRL": "Rail",
    "pRF": "Reflection"
})


dic_return()


# 딕셔너리를 리스트로 변환
records = []
for (risk_code, prompt_code), stats in final_stat_dict.items():
    readable_risk_type = risk_types.get(risk_code, risk_code)  # fallback 처리 포함
    readable_prompt_type = prompt_types.get(prompt_code, prompt_code)  # fallback 처리 포함

    records.append({
        "risk_code": risk_code.strip(),
        "risk_type": readable_risk_type,
        "prompt_type": readable_prompt_type,
        "sum_base_score": stats["sum_base_score"],
        "weighted_mean_score": stats["weighted_mean_score"]
    })

def extract_risk_number(risk_code: str) -> int:
    """문자열에서 숫자만 추출하여 정수로 반환 ('r02' → 2, 'r10' → 10 등)"""
    match = re.search(r"\d+", risk_code)
    return int(match.group()) if match else float('inf')

# ✅ 숫자 기반 정렬 적용
records.sort(key=lambda r: (extract_risk_number(r["risk_code"]), r["prompt_type"]))

# 정렬 결과 확인
for idx, r in enumerate(records[:10]):
    print(f"{idx+1:02d} | {r['risk_code']} | {r['prompt_type']}")

# DataFrame → Pivot (행: prompt_code, 열: risk_code)
df = pd.DataFrame(records)


heatmap_df_weight = df.pivot(index="prompt_type", columns="risk_type", values="weighted_mean_score")

heatmap_df_avg = df.pivot(index="prompt_type", columns="risk_type", values="sum_base_score")
# 🔢 float으로 변환
heatmap_df_weight = heatmap_df_weight.astype(float)
heatmap_df_avg = heatmap_df_avg.astype(float)

# --------------------------
# 탭 구성 시작
# --------------------------
#main_tabs = st.tabs(["📊 Heatmap", "📙 위험 카테고리 분석", "📘 프롬프트 분석", "💬 대화 예시"])

main_tabs = st.tabs(["📊 Heatmap", "📙 위험 카테고리 분석"])

# 📊 Heatmap
with main_tabs[0]:
    st.subheader("📊 위험 점수 Heatmap - 가중평균")

    # NaN을 기준으로 마스킹
    mask_matrix = heatmap_df_weight.isna()

    # 스타일링 함수 생성
    highlight_func = highlight_excluded_rows_factory(excluded_ids)

    fig, ax = plt.subplots(figsize=(20, 6))

    # NaN에만 색 마스크
    sns.heatmap(heatmap_df_weight.astype(float), mask=mask_matrix, annot=True, fmt=".1f", cmap="YlGnBu", vmin=1, vmax=5,linewidths=0.2,linecolor='lightgray', cbar_kws={'label': 'Safety Score', 'shrink': 0.6, 'aspect': 20},annot_kws={"size": 8})
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    excluded_ids = set()  # 지금은 예외 없음

    # 📋 시뮬레이션 데이터 보기 영역
    with st.expander("📋 가중평균 데이터 보기"):
        styled_df = heatmap_df_weight.style.format("{:.2f}").apply(highlight_func, axis=1)
        st.dataframe(styled_df)

    
    st.subheader("📊 위험 점수 Heatmap - 산술평균")

    # NaN을 기준으로 마스킹
    mask_matrix = heatmap_df_avg.isna()

    fig, ax = plt.subplots(figsize=(20, 6))

    # NaN에만 색 마스크
    sns.heatmap(heatmap_df_avg.astype(float), mask=mask_matrix, annot=True, fmt=".1f", cmap="YlGnBu", vmin=1, vmax=5,linewidths=0.2,linecolor='lightgray', cbar_kws={'label': 'Safety Score', 'shrink': 0.6, 'aspect': 20},annot_kws={"size": 8})
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    excluded_ids = set()  # 지금은 예외 없음
        
    # 📋 시뮬레이션 데이터 보기 영역
    with st.expander("📋 산술평균 데이터 보기"):
        styled_df = heatmap_df_avg.style.format("{:.2f}").apply(highlight_func, axis=1)
        st.dataframe(styled_df)


# 📙 위험 카테고리별 분석 (탭)
with main_tabs[1]:
    st.subheader("📙 위험 카테고리별 프롬프트 분석")
    
    # ⚙️ 위험 카테고리별 점수 집계용 딕셔너리
    risk_bar_data = defaultdict(list)

    # 📊 final_stat_dict에서 위험 카테고리별 weighted 점수 수집
    for (risk_code, prompt_code), stats in final_stat_dict.items():
        risk_bar_data[risk_code].append(stats["weighted_mean_score"])

    # 🎯 평균 계산
    risk_score_avg = {
        risk_code: sum(scores) / len(scores)
        for risk_code, scores in risk_bar_data.items()
    }

    # 🧾 시각화용 라벨 및 값 준비
    sorted_risks = sorted(risk_score_avg.items(), key=lambda x: int(x[0][1:]))  # r01 → 1
    x_labels = [f"{risk_code} {risk_types.get(risk_code, '')}" for risk_code, _ in sorted_risks]
    y_scores = [score for _, score in sorted_risks]

    # 📊 Bar Chart 그리기
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x_labels, y_scores, color='skyblue')
    ax.set_xlabel("risk category", fontsize=12)
    ax.set_ylabel("weight_score", fontsize=12)
    ax.set_ylim(0, 5)  # ✅ y축 범위 고정 (0~5)
    plt.xticks(rotation=90)

    st.pyplot(fig)
    # ⛳ df.columns[:8]를 명시적으로 리스트로 변환
    # category_labels = df.columns[:8].tolist()
    # category_tabs = st.tabs(category_labels)

    # for i, tab in enumerate(category_tabs):
    #     category = category_labels[i]
    #     with tab:
    #         st.markdown(f"### 📈 {category}에 대한 프롬프트별 리스크 점수")
    #         category_scores = df[category]
    #         fig2, ax2 = plt.subplots(figsize=(10, 4))
    #         cmap = cm.get_cmap('YlGnBu')
    #         norm = plt.Normalize(1, 5)
    #         colors = cmap(norm(category_scores.values))

    #         sns.barplot(x=category_scores.index, y=category_scores.values, palette=colors, ax=ax2)
    #         ax2.set_ylim(0, 5)
    #         ax2.set_xticklabels(category_scores.index, rotation=0, fontsize=9)
    #         ax2.grid(axis='y', linestyle='--', alpha=0.3)
    #         st.pyplot(fig2)

    #         with st.expander("📋 점수 테이블 보기"):
    #             category_df = pd.DataFrame({
    #                 "Prompt Type": category_scores.index,
    #                 "Score": category_scores.values
    #             })
    #             st.dataframe(category_df.style.format({"Score": "{:.2f}"}))

# 📘 프롬프트별 분석
# with main_tabs[2]:
#     st.subheader("📘 프롬프트 타입별 위험 항목 분석 (탭 기반)")
#     tabs = st.tabs(prompt_types)

    # for i, tab in enumerate(tabs):
    #     prompt_type = prompt_types[i]
    #     with tab:
    #         st.markdown(f"### 📊 {prompt_type} 프롬프트의 위험 카테고리 점수")
    #         prompt_scores = df.loc[prompt_type]
    #         fig, ax = plt.subplots(figsize=(14, 4))
    #         cmap = cm.get_cmap('YlGnBu')
    #         norm = plt.Normalize(1, 5)
    #         colors = cmap(norm(prompt_scores.values))

    #         sns.barplot(x=prompt_scores.index, y=prompt_scores.values, palette=colors, ax=ax)
    #         ax.set_ylim(0, 5)
    #         ax.set_xticklabels(prompt_scores.index, rotation=90, fontsize=8)
    #         ax.grid(axis='y', linestyle='--', alpha=0.3)
    #         st.pyplot(fig)

    #         with st.expander(f"📋 {prompt_type} 점수 테이블"):
    #             st.dataframe(pd.DataFrame({"Risk Category": prompt_scores.index, "Safety Score": prompt_scores.values}).style.format({"Safety Score": "{:.2f}"}))

# 💬 대화 예시
# with main_tabs[3]:
    st.subheader("💬 대화 예시 보기")

    def generate_sample_dialogue():
        return [
            {"user": "이건 무엇인가요?"},
            {"model": "이건 예시 응답입니다. 실제로는 모델이 여기에 답변을 생성합니다."},
            {"user": "좀 더 자세히 알려줄 수 있어?"},
            {"model": "물론이죠. 이 부분에 대해 더 자세히 설명드리겠습니다."}
        ]

    # chat_dataset = {}
    # for category in df.columns:
    #     for prompt in prompt_types:
    #         key = (category, prompt)
    #         chat_dataset[key] = []
    #         for _ in range(random.randint(1, 3)):
    #             dialogue_id = str(uuid.uuid4())[:8]
    #             chat_dataset[key].append((dialogue_id, generate_sample_dialogue()))

    # selected_category_chat = st.selectbox("Select Risk Category", df.columns)
    # selected_prompt_chat = st.selectbox("Select Prompt Type", prompt_types)
    # filtered_chats = chat_dataset.get((selected_category_chat, selected_prompt_chat), [])

    # if filtered_chats:
    #     col1, col2 = st.columns([1, 2])
    #     with col1:
    #         st.markdown("### 🗂️ 대화 목록")
    #         selected_dialogue = None
    #         selected_index = None
    #         for i, (d_id, dialogue) in enumerate(filtered_chats):
    #             with st.container():
    #                 st.markdown(f"""
    #                 <div style='border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px 20px; margin-bottom: 12px; background-color: #fafafa;'>
    #                     <div style='font-size: 16px; font-weight: 600;'>🆔 Dialogue ID: {d_id}</div>
    #                     <div style='font-size: 14px; color: #666;'>💬 Turns: {len(dialogue)} | Index: {i+1}</div>
    #                 </div>
    #                 """, unsafe_allow_html=True)
    #                 if st.button(f"🔍 이 대화 보기", key=f"view_{i}"):
    #                     selected_dialogue = dialogue
    #                     selected_index = i

    #     with col2:
    #         if selected_dialogue:
    #             st.markdown(f"### 💬 선택된 대화 보기 (ID: {filtered_chats[selected_index][0]})")
    #             for turn in selected_dialogue:
    #                 if "user" in turn:
    #                     st.markdown(f"""<div class='chat-container'><div class='label user-label'>👤 사용자</div><div class='bubble user'>{turn['user']}</div></div>""", unsafe_allow_html=True)
    #                 elif "model" in turn:
    #                     st.markdown(f"""<div class='chat-container'><div class='label model-label'>🤖 모델</div><div class='bubble model'>{turn['model']}</div></div>""", unsafe_allow_html=True)
    # else:
    #     st.warning("❗ 선택한 조합에 해당하는 대화가 없습니다.")
