import model.m_mysql as db

def add_user(user_vo):
    sql = 'insert into t_user(user_name, email, login_name, login_pwd, \
           create_date, salary) values(%s, %s, %s, %s, sysdate(), 0.0)'
    params = (user_vo['user_name'], user_vo['email'], \
              user_vo['login_name'], user_vo['login_pwd'])
    user_id, affected_rows = db.insert(sql, params)
    return user_id

def login_user(login_name, login_pwd):
    sql = 'select user_id from t_user where login_name=%s and \
           login_pwd=%s'
    params = (login_name, login_pwd)
    rowcount, rows = db.query(sql, params)
    user_id = 0
    if rowcount>0:
        user_id = rows[0][0]
    return user_id

def get_user_vo(user_id):
    sql = 'select user_name, email, login_name, login_pwd from t_user \
           where user_id=%s'
    params = (user_id)
    rowcount, rows = db.query(sql, params)
    user_vo = {}
    if rowcount>0:
        user_vo['user_name'] = rows[0][0]
        user_vo['email'] = rows[0][1]
        user_vo['login_name'] = rows[0][2]
        user_vo['login_pwd'] = rows[0][3]
    return user_vo


