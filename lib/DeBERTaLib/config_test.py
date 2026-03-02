"""
Unit tests for lib/DeBERTaLib/config.py

This module tests the configuration classes AbsModelConfig and ModelConfig,
including dictionary and JSON serialization/deserialization.
"""

import json
import os
import tempfile
import pytest
from lib.DEBERTaLib.config import AbsModelConfig, ModelConfig


class TestAbsModelConfig:
    """Test suite for AbsModelConfig base class."""

    def test_from_dict_simple_values(self):
        """Test from_dict with simple key-value pairs."""
        json_object = {
            "hidden_size": 768,
            "num_layers": 12,
            "name": "test_model"
        }
        config = AbsModelConfig.from_dict(json_object)

        assert hasattr(config, 'hidden_size')
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.name == "test_model"

    def test_from_dict_nested_dict(self):
        """Test from_dict with nested dictionaries."""
        json_object = {
            "hidden_size": 768,
            "attention": {
                "num_heads": 12,
                "dropout": 0.1
            }
        }
        config = AbsModelConfig.from_dict(json_object)

        assert config.hidden_size == 768
        assert hasattr(config, 'attention')
        assert isinstance(config.attention, AbsModelConfig)
        assert config.attention.num_heads == 12
        assert config.attention.dropout == 0.1

    def test_from_dict_deeply_nested(self):
        """Test from_dict with deeply nested dictionaries."""
        json_object = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 42
                    }
                }
            }
        }
        config = AbsModelConfig.from_dict(json_object)

        assert isinstance(config.level1, AbsModelConfig)
        assert isinstance(config.level1.level2, AbsModelConfig)
        assert isinstance(config.level1.level2.level3, AbsModelConfig)
        assert config.level1.level2.level3.value == 42

    def test_from_dict_empty(self):
        """Test from_dict with empty dictionary."""
        config = AbsModelConfig.from_dict({})
        assert config.__dict__ == {}

    def test_from_dict_single_value(self):
        """Test from_dict with single key-value pair."""
        json_object = {"name": "test"}
        config = AbsModelConfig.from_dict(json_object)
        assert config.name == "test"

    def test_from_dict_special_characters(self):
        """Test from_dict with keys containing special characters."""
        json_object = {
            "key_with_underscore": 1,
            "key-with-dash": 2,
            "key.with.dot": 3,
            "key with space": 4
        }
        config = AbsModelConfig.from_dict(json_object)

        assert config.key_with_underscore == 1
        assert config.key_with_dash == 2  # Converted to valid Python attribute name
        assert config.key_dot_dot == 3  # Note: dots replaced by underscores
        assert hasattr(config, 'key with space')  # Spaces preserved in dict access

    def test_to_dict_simple(self):
        """Test to_dict with simple attributes."""
        config = AbsModelConfig()
        config.hidden_size = 768
        config.num_layers = 12

        result = config.to_dict()

        assert result == {'hidden_size': 768, 'num_layers': 12}
        assert result is not config.__dict__  # Deep copy, not reference

    def test_to_dict_nested_config(self):
        """Test to_dict with nested AbsModelConfig objects."""
        config = AbsModelConfig()
        config.hidden_size = 768
        config.attention = AbsModelConfig()
        config.attention.num_heads = 12

        result = config.to_dict()

        assert result['hidden_size'] == 768
        assert isinstance(result['attention'], dict)
        assert result['attention']['num_heads'] == 12

    def test_to_dict_creates_deep_copy(self):
        """Test that to_dict creates a deep copy, not a reference."""
        config = AbsModelConfig()
        config.value = [1, 2, 3]

        result = config.to_dict()
        result['value'].append(4)

        assert config.value == [1, 2, 3]  # Original unchanged
        assert result['value'] == [1, 2, 3, 4]  # Modified copy

    def test_to_json_string_simple(self):
        """Test to_json_string with simple attributes."""
        config = AbsModelConfig()
        config.name = "test"
        config.value = 42

        json_str = config.to_json_string()
        result = json.loads(json_str)

        assert result == {"name": "test", "value": 42}
        assert json_str.endswith('\n')

    def test_to_json_string_sorted_keys(self):
        """Test that to_json_string sorts keys."""
        config = AbsModelConfig()
        config.z_value = 1
        config.a_value = 2
        config.m_value = 3

        json_str = config.to_json_string()
        result = json.loads(json_str)

        keys = list(result.keys())
        assert keys == ['a_value', 'm_value', 'z_value']

    def test_to_json_string_nested_config(self):
        """Test to_json_string with nested AbsModelConfig."""
        config = AbsModelConfig()
        config.name = "test"
        config.nested = AbsModelConfig()
        config.nested.value = 42

        json_str = config.to_json_string()
        result = json.loads(json_str)

        assert result['name'] == "test"
        assert result['nested'] == {'value': 42}

    def test_from_json_file_valid(self):
        """Test from_json_file with a valid JSON file."""
        json_data = {
            "hidden_size": 768,
            "num_layers": 12,
            "name": "test_model"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            config = AbsModelConfig.from_json_file(temp_path)

            assert config.hidden_size == 768
            assert config.num_layers == 12
            assert config.name == "test_model"
        finally:
            os.unlink(temp_path)

    def test_from_json_file_nested(self):
        """Test from_json_file with nested JSON structure."""
        json_data = {
            "model": {
                "hidden_size": 768,
                "attention": {
                    "num_heads": 12
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            config = AbsModelConfig.from_json_file(temp_path)

            assert isinstance(config.model, AbsModelConfig)
            assert config.model.hidden_size == 768
            assert isinstance(config.model.attention, AbsModelConfig)
            assert config.model.attention.num_heads == 12
        finally:
            os.unlink(temp_path)

    def test_from_json_file_nonexistent(self):
        """Test from_json_file with non-existent file."""
        with pytest.raises(FileNotFoundError):
            AbsModelConfig.from_json_file("/nonexistent/path/to/file.json")

    def test_from_json_file_invalid_json(self):
        """Test from_json_file with invalid JSON content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                AbsModelConfig.from_json_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_from_json_file_with_unicode(self):
        """Test from_json_file with Unicode characters."""
        json_data = {
            "name": "测试模型",
            "description": "中文描述 🚀",
            "emoji": "✅"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            config = AbsModelConfig.from_json_file(temp_path)

            assert config.name == "测试模型"
            assert config.description == "中文描述 🚀"
            assert config.emoji == "✅"
        finally:
            os.unlink(temp_path)

    def test_repr_returns_json_string(self):
        """Test __repr__ returns JSON string representation."""
        config = AbsModelConfig()
        config.name = "test"

        repr_str = repr(config)

        assert isinstance(repr_str, str)
        result = json.loads(repr_str)
        assert result == {"name": "test"}

    def test_round_trip_dict(self):
        """Test serialization to dict and deserialization back to config."""
        original_config = AbsModelConfig()
        original_config.hidden_size = 768
        original_config.num_layers = 12
        original_config.name = "test"

        dict_repr = original_config.to_dict()
        new_config = AbsModelConfig.from_dict(dict_repr)

        assert new_config.__dict__ == original_config.__dict__

    def test_round_trip_json_file(self):
        """Test serialization to JSON file and deserialization back."""
        original_config = AbsModelConfig()
        original_config.hidden_size = 768
        original_config.num_layers = 12

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            f.write(original_config.to_json_string())
            temp_path = f.name

        try:
            new_config = AbsModelConfig.from_json_file(temp_path)

            assert new_config.hidden_size == original_config.hidden_size
            assert new_config.num_layers == original_config.num_layers
        finally:
            os.unlink(temp_path)


class TestModelConfig:
    """Test suite for ModelConfig class."""

    def test_init_default_values(self):
        """Test ModelConfig initialization with default values."""
        config = ModelConfig()

        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.hidden_act == "gelu"
        assert config.intermediate_size == 3072
        assert config.hidden_dropout_prob == 0.1
        assert config.attention_probs_dropout_prob == 0.1
        assert config.max_position_embeddings == 512
        assert config.type_vocab_size == 0
        assert config.initializer_range == 0.02
        assert config.layer_norm_eps == 1e-7
        assert config.padding_idx == 0
        assert config.vocab_size == -1

    def test_init_is_abs_model_config_subclass(self):
        """Test that ModelConfig is a subclass of AbsModelConfig."""
        config = ModelConfig()
        assert isinstance(config, AbsModelConfig)

    def test_set_custom_values(self):
        """Test setting custom values after initialization."""
        config = ModelConfig()
        config.hidden_size = 1024
        config.num_hidden_layers = 24

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24

    def test_to_dict_includes_all_attributes(self):
        """Test that to_dict includes all ModelConfig attributes."""
        config = ModelConfig()
        result = config.to_dict()

        expected_keys = [
            'hidden_size', 'num_hidden_layers', 'num_attention_heads',
            'hidden_act', 'intermediate_size', 'hidden_dropout_prob',
            'attention_probs_dropout_prob', 'max_position_embeddings',
            'type_vocab_size', 'initializer_range', 'layer_norm_eps',
            'padding_idx', 'vocab_size'
        ]

        for key in expected_keys:
            assert key in result

    def test_from_dict_override_defaults(self):
        """Test from_dict overriding default ModelConfig values."""
        json_object = {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16
        }

        config = ModelConfig.from_dict(json_object)

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.hidden_act == "gelu"  # Default preserved

    def test_from_dict_preserves_defaults_for_omitted(self):
        """Test that omitted keys preserve default values."""
        json_object = {
            "hidden_size": 1024
        }

        config = ModelConfig.from_dict(json_object)

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 12  # Default
        assert config.hidden_act == "gelu"  # Default

    def test_round_trip_model_config(self):
        """Test full serialization round-trip for ModelConfig."""
        original = ModelConfig()
        original.hidden_size = 1024
        original.num_hidden_layers = 24

        dict_repr = original.to_dict()
        restored = ModelConfig.from_dict(dict_repr)

        assert restored.hidden_size == original.hidden_size
        assert restored.num_hidden_layers == original.num_hidden_layers

    def test_from_json_file_creates_model_config(self):
        """Test from_json_file creates proper ModelConfig instance."""
        json_data = {
            "hidden_size": 1024,
            "num_hidden_layers": 24
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            config = ModelConfig.from_json_file(temp_path)

            assert isinstance(config, ModelConfig)
            assert isinstance(config, AbsModelConfig)
            assert config.hidden_size == 1024
            assert config.num_hidden_layers == 24
        finally:
            os.unlink(temp_path)

    def test_to_json_string_sorted_model_config(self):
        """Test that to_json_string sorts ModelConfig attributes."""
        config = ModelConfig()
        json_str = config.to_json_string()
        result = json.loads(json_str)

        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_edge_case_negative_values(self):
        """Test ModelConfig with negative values."""
        config = ModelConfig()
        config.hidden_size = -1
        config.layer_norm_eps = -1e-7

        assert config.hidden_size == -1
        assert config.layer_norm_eps == -1e-7

    def test_edge_case_zero_values(self):
        """Test ModelConfig with zero values."""
        config = ModelConfig()
        config.hidden_size = 0
        config.hidden_dropout_prob = 0.0

        assert config.hidden_size == 0
        assert config.hidden_dropout_prob == 0.0

    def test_edge_case_large_values(self):
        """Test ModelConfig with large values."""
        config = ModelConfig()
        config.hidden_size = 10**9
        config.num_hidden_layers = 10**6

        assert config.hidden_size == 10**9
        assert config.num_hidden_layers == 10**6

    def test_edge_case_float_values(self):
        """Test ModelConfig with float precision values."""
        config = ModelConfig()
        config.hidden_dropout_prob = 0.123456789
        config.layer_norm_eps = 1e-10

        assert config.hidden_dropout_prob == 0.123456789
        assert config.layer_norm_eps == 1e-10

    def test_string_attribute_values(self):
        """Test ModelConfig with various string attribute values."""
        config = ModelConfig()
        config.hidden_act = "relu"
        config.hidden_act = "swish"

        assert config.hidden_act == "swish"

    def test_mixed_types_in_dict(self):
        """Test from_dict with mixed value types."""
        json_object = {
            "int_value": 42,
            "float_value": 3.14,
            "str_value": "test",
            "bool_value": True,
            "list_value": [1, 2, 3],
            "none_value": None
        }

        config = ModelConfig.from_dict(json_object)

        assert config.int_value == 42
        assert config.float_value == 3.14
        assert config.str_value == "test"
        assert config.bool_value is True
        assert config.list_value == [1, 2, 3]
        assert config.none_value is None

    def test_empty_string_keys(self):
        """Test from_dict with empty string as key."""
        json_object = {"": "empty_key_value"}

        config = AbsModelConfig.from_dict(json_object)

        # Empty string is a valid attribute name
        assert hasattr(config, '')
        assert getattr(config, '') == "empty_key_value"

    def test_numeric_string_keys(self):
        """Test from_dict with numeric string keys."""
        json_object = {"123": "numeric_key"}

        config = AbsModelConfig.from_dict(json_object)

        assert getattr(config, '123') == "numeric_key"

    def test_config_isolation(self):
        """Test that multiple config instances are isolated."""
        config1 = ModelConfig()
        config2 = ModelConfig()

        config1.hidden_size = 1024
        config2.hidden_size = 512

        assert config1.hidden_size == 1024
        assert config2.hidden_size == 512

    def test_nested_config_isolation(self):
        """Test that nested configs are properly isolated."""
        config1 = AbsModelConfig()
        config1.nested = AbsModelConfig()
        config1.nested.value = 1

        config2 = AbsModelConfig()
        config2.nested = AbsModelConfig()
        config2.nested.value = 2

        assert config1.nested.value == 1
        assert config2.nested.value == 2
