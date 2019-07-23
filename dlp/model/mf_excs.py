import model.m_mysql as db

def get_excs_ques_num(excs_id):
    sql = 'select count(ques_id) from t_excs_ques where excs_id=%s'
    params = (excs_id)
    rowcount, rows = db.query(sql, params)
    if rowcount < 1:
        return 0
    return rows[0][0]

def get_excs_id(stut_id):
    sql = 'select E.excs_id from t_excs E, t_excs_stut ES where E.excs_id=ES.excs_id and sysdate()>=start_date and sysdate()<end_date and ES.stut_id=%s'
    params = (stut_id)
    rowcount, rows = db.query(sql, params)
    if rowcount < 1:
        return 0
    return rows[0][0]
    
def get_excs_ques(excs_id, seq):
    sql = 'select Q.ques_id, Q.ques_type_id from t_excs_ques EQ, t_ques Q where EQ.ques_id=Q.ques_id and excs_id=%s and seq=%s'
    params = (excs_id, seq)
    rowcount, rows = db.query(sql, params)
    if rowcount < 1:
        return 0, 0
    return rows[0][0], rows[0][1]
    
def get_excs_quess(excs_id):
    sql = 'select ques_id, seq from t_excs_ques where excs_id=%s'
    params = (excs_id)
    rowcount, rows = db.query(sql, params)
    return rows

def get_stut_ques(stut_id):
    sql = 'select ques_id from t_stut_ques where stut_id=%s'
    params = (stut_id)
    rowcount, rows = db.query(sql, params)
    return rows

def get_ques_stem_file(ques_id):
    sql = 'select T.tpl_file from t_ques_stem Q, t_tpl T where Q.tpl_id=T.tpl_id and Q.ques_form_id=1 and Q.ques_id=%s'
    params = (ques_id)
    rowcount, rows = db.query(sql, params)
    return rows[0][0]
    
def get_ques_optns(ques_id):
    sql = 'select ques_optn_id, T.tpl_file from t_ques_optn O, t_tpl T where O.tpl_id=T.tpl_id and ques_form_id=1 and ques_id=%s'
    params = (ques_id)
    rowcount, rows = db.query(sql, params)
    return rows
    
def get_stut_ss_ques_ansr(stut_id, excs_id, ques_id):
    ''' 获取单选题答案 '''
    sql = 'select SQA.ques_optn_id from t_stut_ques SQ, t_stut_ques_ansr SQA where SQ.stut_ques_id=SQA.stut_ques_id and SQ.stut_id=%s and SQ.excs_id=%s and SQ.ques_id=%s'
    params = (stut_id, excs_id, ques_id)
    rowcount, rows = db.query(sql, params)
    if rowcount < 1:
        return 0
    return rows[0][0]

def get_ques_stem_param(ques_id):
    sql = 'select TP.param_key, QSP.param_val from t_ques_stem QS, t_ques_stem_param QSP, t_tpl_param TP where QS.ques_stem_id=QSP.ques_stem_id and QSP.tpl_param_id=TP.tpl_param_id and QS.ques_id=%s'
    params = (ques_id)
    rowcount, rows = db.query(sql, params)
    param_dict = {}
    for row in rows:
        param_dict[row[0]] = row[1]
    return param_dict
    
def get_ques_optn_param(ques_optn_id):
    sql = 'select TP.param_key, QOP.param_val from t_ques_optn QO, t_ques_optn_param QOP, t_tpl_param TP where QO.ques_optn_id=QOP.ques_optn_id and QOP.tpl_param_id=TP.tpl_param_id and QO.ques_optn_id=%s'
    params = (ques_optn_id)
    rowcount, rows = db.query(sql, params)
    param_dict = {}
    for row in rows:
        param_dict[row[0]] = row[1]
    return param_dict
    
def get_stut_ques_id(excs_id, ques_id, stut_id):
    sql = 'select stut_ques_id from t_stut_ques where excs_id=%s and ques_id=%s and stut_id=%s'
    params = (excs_id, ques_id, stut_id)
    rowcount, rows = db.query(sql, params)
    if rowcount < 1:
        return 0
    return rows[0][0]
    
def get_ques_type(ques_id):
    sql = 'select Q.ques_type_id, QT.ques_type_name from t_ques Q, t_ques_type QT where Q.ques_type_id=QT.ques_type_id and Q.ques_id=%s'
    params = (ques_id)
    rowcount, rows = db.query(sql, params)
    if rowcount < 1:
        return 0
    return rows[0][0]
    
class MFExcs(object):
    def __init__(self):
        pass
        
    @staticmethod
    def get_ques_ansr(ques_id):
        sql = 'select ques_optn_id, val, seq from t_ques_ansr where ques_id=%s'
        params = (ques_id)
        rowcount, rows = db.query(sql, params)
        if rowcount < 1:
            return []
        recs = []
        for row in rows:
            rec = {'ques_optn_id': row[0], 'val': row[1], 'seq': row[2]}
            recs.append(rec)
        return recs
        
    @staticmethod
    def get_stut_ques_optns(stut_id, ques_id):
        sql = 'select ques_optn_id, val, seq from t_stut_ques_ansr SQA, t_stut_ques SQ where SQA.stut_ques_id=SQ.stut_ques_id and SQ.stut_id=%s and SQ.ques_id=%s'
        params = (stut_id, ques_id)
        rowcount, rows = db.query(sql, params)
        recs = []
        if rowcount >= 1:
            for row in rows:
                rec = {'ques_optn_id': row[0], 'val': row[1], 'seq': row[2]}
                recs.append(rec)
        return recs
        
    @staticmethod
    def get_excs_ques_stut_score(excs_id, ques_id, stut_id):
        sql = 'select score from t_stut_ques where excs_id=%s and ques_id=%s and stut_id=%s'
        params = (excs_id, ques_id, stut_id)
        rowcount, rows = db.query(sql, params)
        if rowcount < 1:
            return 0.0
        return rows[0][0]
        
    @staticmethod
    def get_excs_stut_state_id(excs_id, stut_id):
        sql = 'select state_id from t_excs_stut where excs_id=%s and stut_id=%s'
        params = (excs_id, stut_id)
        rowcount, rows = db.query(sql, params)
        if rowcount < 1:
            return 1
        return rows[0][0]
        
    @staticmethod
    def get_chief_tchr_id(stut_id, major_id):
        sql = 'select CMT.tchr_id from t_class_stut CS, t_class_major CM, t_class_major_tchr CMT where CS.class_id=CM.class_id and CM.class_major_id=CMT.class_major_id and CS.stut_id=%s and CM.major_id=%s and sysdate()>=CS.start_date and sysdate()<CS.end_date;'
        params = (stut_id, major_id)
        rowcount, rows = db.query(sql, params)
        if rowcount < 1:
            return 0
        return rows[0][0]
        
    @staticmethod
    def get_ques_teach_video_url(ques_id, tchr_id):
        sql = 'select QT.ques_teach_name, QT.video_file_id, F.file_url from t_ques_teach QT, t_file F where QT.video_file_id=F.file_id and QT.ques_id=%s and QT.tchr_id=%s';
        params = (ques_id, tchr_id)
        rowcount, rows = db.query(sql, params)
        if rowcount < 1:
            return '';
        rst = {'ques_teach_name': rows[0][0], 'video_file_url': rows[0][2]}
        return rst
        
    def get_ques_expl_url(ques_id):
        return 'http://www.sohu.com'
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

