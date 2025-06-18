import numpy as np
import pandas as pd
import os
import zipfile
import json
import re
from collections import OrderedDict, defaultdict

def get_risk_definitions():
    full_risk_labels = {
        #"r01": "1. Supporting Malicious Organized Groups",
        "r02": "2. Celebrating Suffering",
        "r03": "3. Violent Acts",
        "r04": "4. Depicting Violence",
        #"r05": "5. Weapon Usage & Development",
        #"r06": "6. Military and Warfare",
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
    }
    return full_risk_labels

def normalize_prompt_id(prompt_id):
    # RPemo, RPedu, RPfun 등은 모두 RP로 처리
    if prompt_id.startswith("RP"):
        return "RP"
    return prompt_id

def generate_dataframe_with_exclusions(prompt_types, risk_types, sorted_grouped_data, transpose=False):
    # 📌 평균 점수 집계를 위한 딕셔너리
    risk_prompt_matrix = defaultdict(dict)
    risk_bar_data = defaultdict(list)
    prompt_bar_data = defaultdict(list)

    # 📍 데이터 파싱
    score_sum = defaultdict(lambda: defaultdict(float))
    score_count = defaultdict(lambda: defaultdict(int))

    for key, value in sorted_grouped_data.items():
        # 🔍 정규표현식으로 rxx, pXX 추출
        match = re.match(r"(r\d+)_t\d+_p(\w+)_\d+", key)
        if match:
            risk_id = match.group(1)
            raw_prompt_id = match.group(2)
            prompt_id = normalize_prompt_id(raw_prompt_id)
            score = value.get("avg_score")

            if score is None:
                continue

            # 🏷️ 원래 이름으로 변환
            risk_name = risk_types.get(risk_id, risk_id)
            prompt_name = prompt_types.get(prompt_id, prompt_id)

            # 누적 합산 및 카운트
            score_sum[risk_name][prompt_name] += score
            score_count[risk_name][prompt_name] += 1

            # 바차트용 데이터도 계속 누적
            risk_bar_data[risk_name].append((prompt_name, score))
            prompt_bar_data[prompt_name].append((risk_name, score))
        else:
            print(f"[⚠️ 정규식 미일치] {key}")

    # ✅ 평균 계산하여 risk_prompt_matrix 생성
    for r in score_sum:
        for p in score_sum[r]:
            avg_score = score_sum[r][p] / score_count[r][p]
            risk_prompt_matrix[r][p] = avg_score
        
    return risk_prompt_matrix, risk_bar_data, prompt_bar_data

def style_dataframe(df, column_flags):
    def apply_style(val, flag):
        return 'background-color: #f5f5f5; color: #bbbbbb' if flag == 0 else ''

    styled_df = df.style.apply(
        lambda col: [apply_style(v, column_flags[i]) for i, v in enumerate(col)], axis=0
    ).format("{:.2f}")

    return styled_df

def MC_parsing():
    return 0

def notMC_parsing(file_name):
    prompt_types = {
    "pMC": "Multiple-Choice",
    "pQO":"Q Only",
    "pMS":"Multi-Session",
    "pRP":"Role-Playing",
    "pCT":"Chain of Thought",
    "pEP":"Expert Prompting",
    "pRL":"Rail",
    "pRF":"Reflection",
    "pRPfun":"Role-Playing (Functional)",
    "pRPedu":"Role-Playing (Educational)",
    "pRPemo":"Role-Playing (Emotional)"
    }

    # 원본 파일과 변경할 이름
    original_file = os.path.join('log', f'{file_name}.eval')
    renamed_file = os.path.join('log', f'{file_name}.zip')

    # 압축을 풀 대상 폴더 (현재 폴더 기준)
    extract_to = os.path.join('unzipped', f'{file_name}')

    # .zip으로 확장자 변경한 파일이 없으면 이름바꾸고 압축 해제까지 실행
    if os.path.isfile(original_file) and not os.path.isfile(renamed_file):
        os.rename(original_file, renamed_file)

        # # 압축 해제 실행
        with zipfile.ZipFile(renamed_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"'{renamed_file}' 압축 해제 완료 → 폴더: {extract_to}")

    # summaries.json 파일 경로
    json_path = os.path.join(extract_to, 'summaries.json')

    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print("📄 summaries.json 로드 완료")

    # 결과 리스트 초기화
    grouped_data = defaultdict(lambda: {"input": None, "risk_code": None, "prompt_type": None, "avg_score": None, "scores": []})

    # 데이터 분류
    for item in json_data:
        id_ = item["id"]
        input_ = item["input"]
        epoch = item["epoch"]
        model_score = item.get("scores", {}).get("model_graded_qa", {})

        match = re.search(r"(r\d{2})_.*?(p[^_]+)", id_)
        if not match:
            continue
        risk_code, prompt_code = match.groups()

        grouped_data[id_]["input"] = input_
        grouped_data[id_]["risk_code"] = risk_code # 리스크 카테고리 명을 표시하려면 매핑 작업 추가적으로 필요
        grouped_data[id_]["prompt_type"] = prompt_types[prompt_code]
        grouped_data[id_]["scores"].append({
            "epoch": epoch,
            "model_graded_qa": model_score
        })

    # 정렬: epoch 오름차순
    for entry in grouped_data.values():
        entry["scores"].sort(key=lambda x: x["epoch"])


    # dict의 key 값을 오름차순으로 정리. 지금 필요는 없지만, keys()를 사용할 경우를 대비해서 일단 정렬
    sorted_grouped_data = OrderedDict(
        sorted(grouped_data.items(), key=lambda x: x[0])  # x[0]은 id
    )

    # epoch 별 스코어의 평균 점수(avg_score) 계산
    for key in sorted_grouped_data.keys():
        score_sum=0.0

        if isinstance(sorted_grouped_data[key]['scores'], list):
            for i in range(len(sorted_grouped_data[key]["scores"])):
                if sorted_grouped_data[key]["scores"][i]["model_graded_qa"]["value"] in ["1", "2", "3", "4", "5"]:
                    score_sum += float(sorted_grouped_data[key]["scores"][i]["model_graded_qa"]["value"])
                else:
                    print(f'[GRADE 오류 발생] data id: {key}')
            avg_score = ( score_sum / len(sorted_grouped_data[key]["scores"]) )
            sorted_grouped_data[key]["avg_score"] = avg_score
        else:
            sorted_grouped_data[key]["avg_score"] = sorted_grouped_data[key]["scores"]["model_graded_qa"]["value"]


    # # 터미널에 결과 출력 (요약)
    # for id_, data in sorted_grouped_data.items():
    #     print(f"\nID: {id_}")
    #     print(f"Input: {data['input']}")
    #     print(f"avg_score: {data['avg_score']}")
    #     for score in data["scores"]:
    #         print(f"  Epoch {score['epoch']}: value={score['model_graded_qa'].get('value')}")

    return sorted_grouped_data

    #------------------------------------------------------------------------------------------------------------------------------

    # ✅ 제외된 risk 행에 회색 스타일 적용

def highlight_excluded_rows_factory(excluded_ids):
    excluded_prefixes = tuple(f"{i}." for i in sorted(excluded_ids))

    def highlight_excluded_rows(row):
        if any(str(row.name).startswith(prefix) for prefix in excluded_prefixes):
            return ['background-color: #f5f5f5; color: #bbbbbb'] * len(row)
        return [''] * len(row)

    return highlight_excluded_rows