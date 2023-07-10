from machine_learning_hep.analysis.analyzer import Analyzer

class AnalyzerD0jets(Analyzer):
    species = "analyzer"
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)