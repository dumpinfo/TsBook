import model.m_mysql as db

def get_ann_hyper_params(ann_id):
    sql = 'select hyper_param_id, hyper_param_val from t_ann_hyper_param \
            where ann_id=%s'
    params = (ann_id)
    rowcount, rows = db.query(sql, params)
    if rowcount != 8:
        print('error:')
        return {}
    ahp = {}
    for row in rows:
        ahp[str(row[0])] = row[1]
    return ahp


