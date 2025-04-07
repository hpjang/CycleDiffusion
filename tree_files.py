import os

def print_tree_to_markdown(
    start_path,
    descriptions=None,
    output_file='folder_structure.md',
    max_depth=None,
    ignore_exts={'.txt', '.pt', '.json', '.log', '.out'},
    ignore_names={'.git', '.gitignore', '__pycache__'}
):
    lines = ["# 📁 프로젝트 폴더 구조\n"]

    def recurse(path, indent='', depth=0):
        if max_depth is not None and depth > max_depth:
            return

        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            lines.append(indent + '- ⚠️ [Permission Denied]')
            return

        dirs = []
        files = []

        for name in entries:
            full_path = os.path.join(path, name)
            if name in ignore_names:
                continue
            if os.path.isfile(full_path):
                if any(name.endswith(ext) for ext in ignore_exts):
                    continue
                files.append(name)
            else:
                dirs.append(name)

        # 디렉토리 먼저, 그 다음 파일
        sorted_entries = sorted(dirs) + sorted(files)

        for name in sorted_entries:
            full_path = os.path.join(path, name)
            is_dir = os.path.isdir(full_path)
            icon = '📁' if is_dir else '📄'
            display_name = f"{icon} `{name}`"

            # 설명 붙이기 (→ 기호 포함)
            if descriptions and full_path in descriptions:
                display_name += f" → **{descriptions[full_path]}**"

            lines.append(indent + f"- {display_name}")

            if is_dir:
                recurse(full_path, indent + '  ', depth + 1)

    recurse(start_path)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✅ 트리 구조가 Markdown 파일 '{output_file}'로 저장되었습니다.")



# 💡 설명 예시
descriptions = {
    './VCTK_2F2M': 'VCTK에서 남2여2에 대한 데이터 폴더 (DiffVC 형식)',
    './converted_all': '음성 변환 결과 파일 폴더',
    './converted_all/CycleDiffusion_Inter_gender': '논문에 사용한 최종 변환 파일(1)',
    './converted_all/CycleDiffusion_Intra_gender': '논문에 사용한 최종 변환 파일(2)',
    './converted_all/DiffVC_Inter_gender': '논문에 사용한 최종 변환 파일(3)',
    './converted_all/DiffVC_Intra_gender': '논문에 사용한 최종 변환 파일(4)',
    './make_converted_wav.py': '학습시킨 모델 inference 코드',
    './data_cycle_4speakers_0810.py': '화자 4명에 대한 데이터셋 코드',
    './data_cycle_4speakers_one_hot_0822.py': '화자 4명에 대한 원핫 인코딩 데이터셋 코드 (화자 인코더 사용 x)',
    './data.py': '기존 DiffVC 코드',
    './get_avg_mels.ipynb': '기존 DiffVC 코드',
    './inference.ipynb': '기존 DiffVC 코드',
    './params.py': '기존 DiffVC 코드',
    './train_dec.py': '기존 DiffVC 코드',
    './train_enc.py': '기존 DiffVC 코드',
    './utils.py': '기존 DiffVC 코드',
    './example': '기존 DiffVC 폴더',
    './filelists': '기존 DiffVC 폴더',
    './hifi-gan': '기존 DiffVC 폴더',
    './logs_enc': '기존 DiffVC 폴더',
    './logs_enc_2speakers': '2화자로만 인코더 학습시킨 모델 weight',
    './model': '모델 코드',
    './get_mels_embeds_HEE.py': 'mels 및 embeds 파일 생성 코드',
    './get_textgrids.py': 'textgrids 파일 생성 코드',
    './loss_graph_copy.py': 'loss 그래프 생성 코드',
    './project_structure.md': '코드 설명 파일',
    './README.md': 'Cycle Diffsuion 논문 관련 README 파일',
    './get_textgrids.py': 'textgrids 파일 생성코드',
    './train_enc_2speakers.py': '2명의 화자로 인코더 학습 코드',
    './tree_files.py': '코드 구조 설명 md 파일 생성 코드',
    './real_last_cycle_train_dec_4speakers_iii3_cycle6_from_50.py': '최종 CycleDiffusion 학습 코드',
    './checkpts': '기존 사전학습 모델 ckpt 폴더 (DiffVC)',
    './calculate': 'mcd 측정 코드 및 결과 파일들',
    './calculate/cal_pymcd.py': 'mcd 측정 코드',
    './calculate/make_json.py': 'mcd 값들 저장될 디폴트 json파일 생성 코드',
    './real_last_cycle_train_dec_4speakers_iii3_cycle6_from_50': '학습한 모델 weight 저장 폴더',
}

# ▶️ 실행
print_tree_to_markdown(
    './',
    descriptions=descriptions,
    output_file='project_structure.md',
    max_depth=1,
    ignore_exts={'.txt', '.pt', '.json', '.log', '.out'},
    ignore_names={'.git', '.gitignore', '__pycache__'}
)
