import model.m_mysql as db

def add_samples(user_id):
    sql = 'insert into t_samples(user_id, create_date, labeled) \
           values(%s, sysdate(), 0)'
    params = (user_id)
    samples_id, affected_rows = db.insert(sql, params)
    return samples_id


