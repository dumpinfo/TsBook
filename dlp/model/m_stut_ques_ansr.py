import model.m_mysql as db

class MStutQuesAnsr(object):
    def __init__(self):
        pass
        
    @staticmethod
    def submit_ques_ansr(stut_ques_id, ques_optn_id, val, seq):
        sql = 'insert into t_stut_ques_ansr(stut_ques_id, ques_optn_id, val, seq) values(%s, %s, %s, %s)'
        params = (stut_ques_id, ques_optn_id, val, seq)
        stut_ques_ansr_id, affected_rows = db.insert(sql, params)
        return stut_ques_ansr_id
        
    @staticmethod
    def delete_ques_ansr(stut_ques_id):
        ''' 删除单选题之前的选项 '''
        sql = 'delete from t_stut_ques_ansr where stut_ques_id=%s'
        params = (stut_ques_id)
        pk, affected_rows = db.delete(sql, params)
        return affected_rows
        
    @staticmethod
    def delete_ques_optn_ansr(stut_ques_id, ques_optn_id):
        ''' 多选题去掉之前的选项 '''
        pass
        
    @staticmethod
    def update_ques_optn_ansr(stut_ques_id, ques_optn_id, val):
        ''' 更新填空题和简答题的答案 '''
        pass