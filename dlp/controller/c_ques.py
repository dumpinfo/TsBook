import sys
sys.path.append('./lib/cherrypy')
sys.path.append('./lib/jinja')
import cherrypy
import json
from jinja2 import Template
import jinja2 as jinja2
import conf.web_conf as web_conf
import app_global as ag
import model.mf_excs as excs
from model.mf_excs import MFExcs as MFExcs
from model.m_stut_ques_ansr import MStutQuesAnsr as MStutQuesAnsr
from model.m_excs_stut import MExcsStut as MExcsStut
from model.m_stut_ques import MStutQues as MStutQues

class CQues(object):
    exposed = True
    def __init__(self):
        self.web_dir = ag.web_dir
        
    @staticmethod
    def get_ques_html(req_args):
        ''' 获取题目的HTML内容 '''
        params = req_args['kwargs']
        print('生成问题页面HTML内容:{0}'.format(params))
        stut_id = params['stut_id']
        ques_seq = params['ques_seq']
        major_id = params['major_id']
        if 'excs_id' in params:
            excs_id = int(params['excs_id'])
        else:
            excs_id = excs.get_excs_id(stut_id)
        if excs_id < 1:
            resp = {}
            resp['status'] = 'Ok'
            resp['excs_id'] = '0'
            resp['ques_num'] = '0'
            resp['html'] = ''
            return resp
        
        ques_num = excs.get_excs_ques_num(excs_id)
        ques_id, ques_type_id = excs.get_excs_ques(excs_id, ques_seq)
        
        tpl_loader = jinja2.FileSystemLoader(searchpath=ag.jinja2_searchpath)
        tpl_env = jinja2.Environment(loader=tpl_loader)
        ques_stem_tpl_file = excs.get_ques_stem_file(ques_id)
        ques_stem_file = '{0}{1}'.format(ag.resources_dir, ques_stem_tpl_file)
        ques_stem_tpl = tpl_env.get_template(ques_stem_file)
        ques_stem_page_params = {}
        ques_stem_dict = excs.get_ques_stem_param(ques_id)
        for key, val in ques_stem_dict.items():
            ques_stem_page_params[key] = val
        ques_stem_page_params['ques_num'] = ques_num
        ques_stem_page_params['ques_seq'] = ques_seq
        ques_stem_page_params['ques_expl_url'] = MFExcs.get_ques_expl_url(ques_id)
        ques_stem_page_params['ques_expl_disp'] = 'display: none;'
        stut_ques_score = MFExcs.get_excs_ques_stut_score(excs_id, ques_id, stut_id)
        excs_stut_state_id = MFExcs.get_excs_stut_state_id(excs_id, stut_id)
        if excs_stut_state_id != 1:
            ques_stem_page_params['ques_expl_disp'] = 'display: block;'
            ques_teach_name, ques_teach_url = CQues.get_ques_teach_url(major_id, ques_id, stut_id)
            ques_stem_page_params['ques_teach_url'] = ques_teach_url
            if stut_ques_score > 3.0:
                ques_stem_page_params['ques_result_png'] = 'right.png'
                ques_stem_page_params['ques_result_display'] = 'block'
            else:
                ques_stem_page_params['ques_result_png'] = 'wrong.png'
                ques_stem_page_params['ques_result_display'] = 'block'
        html_str = ques_stem_tpl.render(ques_stem_page_params)
        html_str += '<div class="weui-cells weui-cells_radio">'
        optns = excs.get_ques_optns(ques_id)
        for optn in optns:
            optn_tpl = tpl_env.get_template(ag.resources_dir + optn[1])
            stut_ques_optn_id = excs.get_stut_ss_ques_ansr(stut_id, excs_id, ques_id)
            optn_param_dict = excs.get_ques_optn_param(optn[0])
            if stut_ques_optn_id == optn[0]:
                check_status = 'checked=true'
            else:
                check_status = ''
            optn_page_param = {}
            for key, val in optn_param_dict.items():
                optn_page_param[key] = val
            optn_page_param['optn_id'] = 'o_{0}_{1}_{2}_{3}'.format(stut_id, excs_id, ques_id, optn[0])
            if excs_stut_state_id != 1:
                optn_page_param['disabled_status'] = 'disabled=true'
            else:
                optn_page_param['disabled_status'] = ''
            optn_page_param['checked_status'] = check_status
            optn_page_param['optn_x_id'] = optn[0]
            html_str += optn_tpl.render(optn_page_param)
        html_str += '</div>'
        fo = open('{0}tpl/qs_{1}.js'.format(ag.resources_dir, ques_type_id), 'r', encoding='utf-8')
        try:
            js = fo.read()
        finally:
            fo.close()
        html_str += js
    
        resp = {}
        resp['status'] = 'Ok'
        resp['excs_id'] = excs_id
        resp['ques_num'] = ques_num
        resp['html'] = html_str
        return resp
        
    @staticmethod
    def submit_optn(req_args):
        ''' 获取题目的HTML内容 '''
        json_obj = req_args['kwargs']['json_obj']
        stut_id = json_obj['stut_id']
        excs_id = json_obj['excs_id']
        ques_id = json_obj['ques_id']
        ques_optn_id = json_obj['optn_id']
        stut_ques_id = excs.get_stut_ques_id(excs_id, ques_id, stut_id)
        ques_type_id = excs.get_ques_type(ques_id)
        if 1 == ques_type_id:
            MStutQuesAnsr.delete_ques_ansr(stut_ques_id)
            stut_ques_ansr_id = MStutQuesAnsr.submit_ques_ansr(stut_ques_id, ques_optn_id, 'Y', 1)
            MStutQues.update_do_date(excs_id, ques_id, stut_id)
        resp = {}
        resp['status'] = 'Ok'
        resp['stut_ques_ansr_id'] = stut_ques_ansr_id
        return resp
        
    @staticmethod
    def submit_excs_ajax(req_args):
        stut_id = int(req_args['kwargs']['stut_id'])
        excs_id = int(req_args['kwargs']['excs_id'])
        CQues.submit_excs(excs_id, stut_id)
        resp = {}
        resp['status'] = 'Ok'
        return resp
        
    @staticmethod
    def submit_excs(excs_id, stut_id):
        ''' 学生按交卷按钮时触发的动作 '''
        MExcsStut.update_excs_stut_state(excs_id, stut_id)
        print('#########################  e:{0}; s:{1}'.format(excs_id, stut_id))
        quess = excs.get_excs_quess(excs_id)
        for ques in quess:
            ques_id = ques[0]
            ques_optns = CQues.get_ques_ansr(ques[0])
            ques_type_id = excs.get_ques_type(ques_id)
            stut_ques_optns = MFExcs.get_stut_ques_optns(stut_id, ques_id)
            print('stut_ques_optns:[{0}] vs ansr:[{1}]'.format(stut_ques_optns, ques_optns))
            if 1 == ques_type_id:
                if len(stut_ques_optns)>0 and ques_optns[0]['ques_optn_id'] == stut_ques_optns[0]['ques_optn_id']:
                    MStutQues.judge_stut_ques(excs_id, ques_id, stut_id, 5.0)
                else:
                    MStutQues.judge_stut_ques(excs_id, ques_id, stut_id, 0.0)
    
    @staticmethod
    def get_ques_ansr(ques_id):
        return MFExcs.get_ques_ansr(ques_id)
        
    @staticmethod
    def get_ques_teach_url_ajax(req_args):
        ''' 获取题目讲解链接 '''
        params = req_args['kwargs']
        stut_id = params['stut_id']
        major_id = params['major_id']
        ques_id = params['ques_id']
        ques_teach_name, ques_teach_url = CQues.get_ques_teach_url(major_id, ques_id, stut_id)
        resp = {}
        resp['status'] = 'Ok'
        resp['ques_teach_url'] = ques_teach_url
        resp['ques_teach_name'] = ques_teach_name
        return resp
        
    @staticmethod
    def get_ques_teach_url(major_id, ques_id, stut_id):
        # 获取学生的首席老师
        tchr_id = MFExcs.get_chief_tchr_id(stut_id, major_id)
        print('tchr_id={0}'.format(tchr_id))
        # 获取讲解视频
        rec = MFExcs.get_ques_teach_video_url(ques_id, tchr_id)
        # 返回结果
        return rec['ques_teach_name'], rec['video_file_url']
    
    @staticmethod
    def test():
        req_args = {'args': (), 'kwargs':{'stut_id': '2', 'major_id': '8', 'ques_id': '1'}}
        resp = CQues.get_ques_teach_url_ajax(req_args)
        print(resp)

