def yaml_to_namespace(yaml_file):
    with open(yaml_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    config_namespace = SimpleNamespace(**config_dict)
    return config_namespace