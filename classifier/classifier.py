import importlib
from setting import opt

def run_classifier():
    classifier_module_name = f"classifier{opt.classify_index}"

    classifier_module = importlib.import_module(classifier_module_name)

    if hasattr(classifier_module, 'main'):
        classifier_module.main()
    else:
        print(f"Module {classifier_module_name} does not have a main function.")

if __name__ == "__main__":
    run_classifier()
