import json
from apps.des.recommend_engine import RecommendEngine as RecommendEngine

class CRecommendEngine(object):
    @staticmethod
    def run_recommend_engine():
        re = RecommendEngine()
        re.run()
        
    @staticmethod
    def save_model_to_db(mu, nu):
        pass
        
    @staticmethod
    def test():
        CRecommendEngine.run_recommend_engine()