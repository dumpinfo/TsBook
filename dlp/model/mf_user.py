import model.m_mysql as db
    
class MFUser(object):
        
    @staticmethod
    def login_user(login_name, login_pwd):
        sql = 'select user_id, user_name, role_id from t_user where login_name=%s and login_pwd=%s'
        params = (login_name, login_pwd)
        rowcount, rows = db.query(sql, params)
        if rowcount < 1:
            return 0, '', 0
        return rows[0][0], rows[0][1], rows[0][2]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

