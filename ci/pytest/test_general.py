from machine_learning_hep.io import parse_yaml

YAML_PATH = "ci/pytest/yaml.yaml"

def test_yaml():
    assert isinstance(parse_yaml(YAML_PATH), dict)
