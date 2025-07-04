from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from config import Config

class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_random_forest(params=None):
        """创建随机森林模型"""
        if params is None:
            params = Config.RF_PARAMS
        return RandomForestClassifier(**params)
    
    @staticmethod
    def create_svm(params=None):
        """创建支持向量机模型"""
        if params is None:
            params = Config.SVM_PARAMS
        return svm.SVC(**params)
    
    @staticmethod
    def create_mlp(params=None):
        """创建多层感知机模型"""
        if params is None:
            params = Config.ANN_PARAMS
        return MLPClassifier(**params)
    
    @staticmethod
    def create_gradient_boosting(params=None):
        """创建梯度提升模型"""
        if params is None:
            params = getattr(Config, 'GB_PARAMS', {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            })
        return GradientBoostingClassifier(**params)
    
    @staticmethod
    def create_ada_boost(params=None):
        """创建AdaBoost模型"""
        if params is None:
            params = getattr(Config, 'ADA_PARAMS', {
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': 42
            })
        return AdaBoostClassifier(**params)
    
    @staticmethod
    def create_logistic_regression(params=None):
        """创建逻辑回归模型"""
        if params is None:
            params = getattr(Config, 'LR_PARAMS', {
                'random_state': 42,
                'max_iter': 1000
            })
        return LogisticRegression(**params)
    
    @staticmethod
    def create_knn(params=None):
        """创建K近邻模型"""
        if params is None:
            params = getattr(Config, 'KNN_PARAMS', {
                'n_neighbors': 5,
                'weights': 'uniform'
            })
        return KNeighborsClassifier(**params)
    
    @staticmethod
    def create_naive_bayes(params=None):
        """创建朴素贝叶斯模型"""
        if params is None:
            params = getattr(Config, 'NB_PARAMS', {})
        return GaussianNB(**params)
    
    @staticmethod
    def create_decision_tree(params=None):
        """创建决策树模型"""
        if params is None:
            params = getattr(Config, 'DT_PARAMS', {
                'random_state': 42,
                'max_depth': 10
            })
        return DecisionTreeClassifier(**params)
    
    @staticmethod
    def create_lda(params=None):
        """创建线性判别分析模型"""
        if params is None:
            params = getattr(Config, 'LDA_PARAMS', {})
        return LinearDiscriminantAnalysis(**params)
    
    @staticmethod
    def create_extra_trees(params=None):
        """创建极端随机树模型"""
        if params is None:
            params = getattr(Config, 'ET_PARAMS', {
                'n_estimators': 100,
                'random_state': 42
            })
        return ExtraTreesClassifier(**params)
    
    @staticmethod
    def create_all_models():
        """创建所有模型并返回模型列表"""
        models = [
            {
                "label": "Model_RF",
                "model": ModelFactory.create_random_forest(),
                "name": "Random Forest"
            },
            {
                "label": "Model_SVM", 
                "model": ModelFactory.create_svm(),
                "name": "Support Vector Machine"
            },
            {
                "label": "Model_ANN",
                "model": ModelFactory.create_mlp(),
                "name": "Artificial Neural Network"
            },
            {
                "label": "Model_GB",
                "model": ModelFactory.create_gradient_boosting(),
                "name": "Gradient Boosting"
            },
            {
                "label": "Model_ADA",
                "model": ModelFactory.create_ada_boost(),
                "name": "AdaBoost"
            },
            {
                "label": "Model_LR",
                "model": ModelFactory.create_logistic_regression(),
                "name": "Logistic Regression"
            },
            {
                "label": "Model_KNN",
                "model": ModelFactory.create_knn(),
                "name": "K-Nearest Neighbors"
            },
            {
                "label": "Model_NB",
                "model": ModelFactory.create_naive_bayes(),
                "name": "Naive Bayes"
            },
            {
                "label": "Model_DT",
                "model": ModelFactory.create_decision_tree(),
                "name": "Decision Tree"
            },
            {
                "label": "Model_LDA",
                "model": ModelFactory.create_lda(),
                "name": "Linear Discriminant Analysis"
            },
            {
                "label": "Model_ET",
                "model": ModelFactory.create_extra_trees(),
                "name": "Extra Trees"
            }
        ]
        
        return models
    
    @staticmethod
    def get_model_by_name(name):
        """根据名称获取特定模型"""
        name = name.lower()
        model_mapping = {
            'rf': ModelFactory.create_random_forest,
            'random_forest': ModelFactory.create_random_forest,
            'svm': ModelFactory.create_svm,
            'support_vector_machine': ModelFactory.create_svm,
            'mlp': ModelFactory.create_mlp,
            'ann': ModelFactory.create_mlp,
            'neural_network': ModelFactory.create_mlp,
            'gb': ModelFactory.create_gradient_boosting,
            'gradient_boosting': ModelFactory.create_gradient_boosting,
            'ada': ModelFactory.create_ada_boost,
            'adaboost': ModelFactory.create_ada_boost,
            'lr': ModelFactory.create_logistic_regression,
            'logistic_regression': ModelFactory.create_logistic_regression,
            'knn': ModelFactory.create_knn,
            'k_nearest_neighbors': ModelFactory.create_knn,
            'nb': ModelFactory.create_naive_bayes,
            'naive_bayes': ModelFactory.create_naive_bayes,
            'dt': ModelFactory.create_decision_tree,
            'decision_tree': ModelFactory.create_decision_tree,
            'lda': ModelFactory.create_lda,
            'linear_discriminant_analysis': ModelFactory.create_lda,
            'et': ModelFactory.create_extra_trees,
            'extra_trees': ModelFactory.create_extra_trees
        }
        
        if name in model_mapping:
            return model_mapping[name]()
        else:
            available_models = list(set([key.replace('_', ' ').title() for key in model_mapping.keys()]))
            raise ValueError(f"未知的模型名称: {name}. 可用模型: {', '.join(available_models)}")
    
    @staticmethod
    def get_available_models():
        """获取所有可用模型名称"""
        return [
            'Random Forest', 'Support Vector Machine', 'Artificial Neural Network',
            'Gradient Boosting', 'AdaBoost', 'Logistic Regression',
            'K-Nearest Neighbors', 'Naive Bayes', 'Decision Tree',
            'Linear Discriminant Analysis', 'Extra Trees'
        ]