import model.m_mysql as db

class MExcsStut(object):
    def __init__(self):
        pass
        
    @staticmethod
    def update_excs_stut_state(excs_id, stut_id):
        ''' 删除单选题之前的选项 '''
        state_id = 2
        sql = 'update t_excs_stut set state_id=%s where excs_id=%s and stut_id=%s'
        params = (state_id, excs_id, stut_id)
        pk, affected_rows = db.update(sql, params)
        affected_rows = 1
        return affected_rows