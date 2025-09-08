# preprocess_ml1m.py
import argparse, os, csv

# ML-1M의 age 카테고리(공식 정의)
AGE_BUCKETS = [1, 18, 25, 35, 45, 50, 56]

def build_user_table(raw_dir):
    """users.dat -> (user_id -> (age_idx, gender_idx, job_idx, zip3_idx)), zip3_vocab"""
    user_rows = {}
    zip3_vocab = {}
    nxt = 0

    users_path = os.path.join(raw_dir, 'users.dat')
    with open(users_path, encoding='latin-1') as f:
        for line in f:
            uid, gender, age, job, zipc = line.strip().split('::')
            uid = int(uid)

            # gender: M=0 / F=1 (이진 인덱스)
            gender_idx = 0 if gender == 'M' else 1

            # age: ML-1M 범주를 인덱스화
            age_val = int(age)
            try:
                age_idx = AGE_BUCKETS.index(age_val)
            except ValueError:
                # 혹시 모를 예외값은 가장 가까운 버킷으로 보정
                closest = min(range(len(AGE_BUCKETS)), key=lambda i: abs(AGE_BUCKETS[i]-age_val))
                age_idx = closest

            job_idx = int(job)

            # zip3: 앞 3자리 → vocab 인덱스
            z3 = (zipc[:3] if len(zipc) >= 3 else zipc).strip()
            if z3 not in zip3_vocab:
                zip3_vocab[z3] = nxt
                nxt += 1
            zip3_idx = zip3_vocab[z3]

            user_rows[uid] = (age_idx, gender_idx, job_idx, zip3_idx)

    return user_rows, zip3_vocab


def write_inter_file(raw_dir, out_dir):
    """ratings.dat -> ml-1m.inter (user_id, item_id, timestamp)"""
    os.makedirs(out_dir, exist_ok=True)
    inter_path = os.path.join(out_dir, 'ml-1m.inter')

    with open(inter_path, 'w', newline='', encoding='utf-8') as wf:
        w = csv.writer(wf, delimiter='\t')
        w.writerow(['user_id:token', 'item_id:token', 'timestamp:float'])

        ratings_path = os.path.join(raw_dir, 'ratings.dat')
        with open(ratings_path, encoding='latin-1') as f:
            for line in f:
                uid, mid, rating, ts = line.strip().split('::')
                # rating은 사용하지 않음 (implicit)
                w.writerow([uid, mid, ts])

    return inter_path


def write_user_file(user_rows, out_dir):
    """user_rows -> ml-1m.user"""
    user_path = os.path.join(out_dir, 'ml-1m.user')
    with open(user_path, 'w', newline='', encoding='utf-8') as wf:
        w = csv.writer(wf, delimiter='\t')
        w.writerow(['user_id:token', 'age_idx:token', 'gender_idx:token', 'job_idx:token', 'zip3_idx:token'])
        for uid in sorted(user_rows):
            age_idx, gender_idx, job_idx, zip3_idx = user_rows[uid]
            w.writerow([uid, age_idx, gender_idx, job_idx, zip3_idx])
    return user_path


def write_item_file(raw_dir, out_dir, with_meta=False):
    """
    movies.dat가 있으면:
      - with_meta=False: 최소 스키마(item_id만)
      - with_meta=True : item_id, title, genres까지 저장 (RecBole에서 load_col.item에 맞춰야 함)
    movies.dat가 없으면 ratings.dat에서 item_id만 수집
    """
    item_path = os.path.join(out_dir, 'ml-1m.item')

    movies_path = os.path.join(raw_dir, 'movies.dat')
    if os.path.exists(movies_path):
        if with_meta:
            # 확장 스키마(원하면 사용) — config에서 item 컬럼도 로드해야 함
            with open(item_path, 'w', newline='', encoding='utf-8') as wf:
                w = csv.writer(wf, delimiter='\t')
                w.writerow(['item_id:token', 'title:token', 'genres:token_seq'])
                with open(movies_path, encoding='latin-1') as f:
                    for line in f:
                        mid, title, genres = line.strip().split('::')
                        # genres는 |로 구분되어 있음 → token_seq로 그대로 저장
                        w.writerow([mid, title, genres])
        else:
            # 최소 스키마(아이템 ID만) — RecBole가 가장 안정적으로 읽음
            with open(item_path, 'w', newline='', encoding='utf-8') as wf:
                w = csv.writer(wf, delimiter='\t')
                w.writerow(['item_id:token'])
                with open(movies_path, encoding='latin-1') as f:
                    for line in f:
                        mid = line.strip().split('::', 1)[0]
                        w.writerow([mid])
    else:
        # movies.dat이 없다면 ratings.dat에서 item_id만 unique 수집
        ids = set()
        ratings_path = os.path.join(raw_dir, 'ratings.dat')
        with open(ratings_path, encoding='latin-1') as f:
            for line in f:
                _, mid, _, _ = line.strip().split('::')
                ids.add(mid)
        with open(item_path, 'w', newline='', encoding='utf-8') as wf:
            w = csv.writer(wf, delimiter='\t')
            w.writerow(['item_id:token'])
            for mid in sorted(ids, key=lambda x: int(x)):
                w.writerow([mid])

    return item_path


def main(raw_dir, out_dir, with_item_meta=False):
    os.makedirs(out_dir, exist_ok=True)

    # 1) user 테이블
    user_rows, zip3_vocab = build_user_table(raw_dir)

    # 2) inter / user / item 파일 생성
    inter_path = write_inter_file(raw_dir, out_dir)
    user_path  = write_user_file(user_rows, out_dir)
    item_path  = write_item_file(raw_dir, out_dir, with_meta=with_item_meta)

    # 안내 출력
    print(f'[OK] wrote:\n- {inter_path}\n- {user_path}\n- {item_path}')
    print(f'zip3_classes = {len(zip3_vocab)}  # config_ml.yaml 의 ml1m_zip3_classes 로 설정하세요')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True, help='원본 ML-1M (users.dat, ratings.dat, movies.dat) 폴더')
    ap.add_argument('--out_dir', required=True, help='전처리 결과(.inter/.user/.item) 저장 폴더')
    ap.add_argument('--with_item_meta', action='store_true',
                    help='movies.dat에서 title/genres까지 .item에 포함 (config.load_col.item도 맞춰야 함)')
    args = ap.parse_args()
    main(args.raw_dir, args.out_dir, with_item_meta=args.with_item_meta)
