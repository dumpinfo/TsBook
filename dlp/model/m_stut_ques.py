import model.m_mysql as db

class MStutQues(object):
    def __init__(self):
        pass
        
    
    def update_do_date(excs_id, ques_id, stut_id):
        sql = 'update t_stut_ques set do_date=sysdate() where excs_id=%s and ques_id=%s and stut_id=%s'
        params = (excs_id, ques_id, stut_id)
        affected_rows = db.update(sql, params)
        
    @staticmethod
    def judge_stut_ques(excs_id, ques_id, stut_id, score):
        sql = 'update t_stut_ques set score=%s where excs_id=%s and ques_id=%s and stut_id=%s'
        params = (score, excs_id, ques_id, stut_id)
        affected_rows = db.update(sql, params)
        
        