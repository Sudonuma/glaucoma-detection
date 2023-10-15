from src.trainer import evaluate_model

def test_evaluate_model(test_options, logger):
    
    accuracy, precision, recall, f1, auc = evaluate_model(test_options, logger)

    assert isinstance(accuracy, float)  
    assert isinstance(precision, float)  
    assert isinstance(recall, float)  
    assert isinstance(f1, float)  
    assert isinstance(auc, float)  
