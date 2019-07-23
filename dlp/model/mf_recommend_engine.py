import numpy as np
import model.m_mysql as db
    
class MFRecommendEngine(object):
    @staticmethod
    def load_dataset():
        # 求出知识点数量
        n = MFRecommendEngine.get_knpt_num()
        # 取出所有学生
        nu = MFRecommendEngine.get_stut_num()
        # 取出所有问题
        nm = MFRecommendEngine.get_ques_num()
        # 取出学生答案
        ph = np.zeros(shape=(nm, nu), dtype=np.float32)
        r = np.ones(shape=(nm, nu), dtype=np.int32)
        stut_ids = MFRecommendEngine.get_stut_ids()
        ques_ids = MFRecommendEngine.get_ques_ids()
        for row, ques_id in enumerate(ques_ids):
            for col, stut_id in enumerate(stut_ids):
                score, do_date = MFRecommendEngine.get_stut_ques_info(stut_id, ques_id)
                if do_date:
                    ph[row][col] = score
                    r[row][col] = 1
                else:
                    ph[row][col] = -1
                    r[row][col] = 0
        return n, nu, nm, ph, r
        
    @staticmethod
    def get_knpt_num():
        sql = 'select count(knpt_id) from t_knpt'
        params = ()
        rc, rows = db.query(sql, params)
        if rc < 1:
            return 0
        return rows[0][0]
        
    @staticmethod
    def get_stut_num():
        sql = 'select count(stut_id) from t_stut'
        params = ()
        rowcount, rows = db.query(sql, params)
        if rowcount < 1:
            return 0
        return rows[0][0]
        
    @staticmethod
    def get_ques_num():
        sql = 'select count(ques_id) from t_ques'
        params = ()
        rowcount, rows = db.query(sql, params)
        if rowcount < 1:
            return 0
        return rows[0][0]
        
    @staticmethod
    def get_raw_data():
        stut_ids = MFRecommendEngine.get_stut_ids()
        ques_ids = MFRecommendEngine.get_ques_ids()
        for stut_id in stut_ids:
            for ques_id in ques_ids:
                print('{0}:{1}'.format(stut_id, ques_id))
        return stut_ids
        
    @staticmethod
    def get_stut_ids():
        sql = 'select stut_id from t_stut'
        params = ()
        rc, rows = db.query(sql, params)
        if rc < 1:
            return []
        return rows
        
    @staticmethod
    def get_ques_ids():
        sql = 'select ques_id from t_ques'
        params = ()
        rc, rows = db.query(sql, params)
        if rc < 1:
            return []
        return rows
        
    @staticmethod
    def get_stut_ques_info(stut_id, ques_id):
        sql = 'select score, do_date from t_stut_ques where stut_id=%s and ques_id=%s order by stut_ques_id desc'
        params = (stut_id, ques_id)
        rc, rows = db.query(sql, params)
        if rc < 1:
            return 0.0, None
        score = 0.0
        if rows[0][0] is None:
            score = 0.0
        else:
            score = 5.0-float(rows[0][0])
        do_date = rows[0][1]
        return score, do_date
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

