import model.m_mysql as db

def get_ann_version(ann_id):
    sql = 'select ann_version_id, ann_version_name, start_date, \
            end_date from t_ann_version where ann_id=%s and \
            ann_version_id=(select max(ann_version_id) \
            from t_ann_version where ann_id=%s)'
    params = (ann_id, ann_id)
    rowcount, rows = db.query(sql, params)
    ann_version_info = {}
    ann_version_info['ann_version_id'] = rows[0][0]
    ann_version_info['ann_version_name'] = rows[0][1]
    ann_version_info['start_date'] = rows[0][2]
    ann_version_info['end_date'] = rows[0][3]
    return ann_version_info


