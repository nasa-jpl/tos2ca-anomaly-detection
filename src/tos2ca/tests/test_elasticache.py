import pickle
import pytest

from tos2ca.database.elasticache import serialize_dict, split_bytes


class TestSerializeDict:
    def test_returns_bytes(self):
        result = serialize_dict({"key": "value"})
        assert isinstance(result, bytes)

    def test_roundtrip(self):
        data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        result = serialize_dict(data)
        assert pickle.loads(result) == data

    def test_empty_dict(self):
        result = serialize_dict({})
        assert pickle.loads(result) == {}

    def test_uses_highest_protocol(self):
        data = {"x": 42}
        result = serialize_dict(data)
        expected = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        assert result == expected


class TestSplitBytes:
    CHUNK_SIZE = 100 * 1024 * 1024  # 100 MB

    def test_small_data_returns_single_chunk(self):
        data = b"hello world"
        chunks = split_bytes(data)
        assert len(chunks) == 1
        assert chunks[0] == data

    def test_empty_bytes_returns_empty_list(self):
        chunks = split_bytes(b"")
        assert chunks == []

    def test_exact_chunk_size_returns_one_chunk(self):
        data = b"x" * self.CHUNK_SIZE
        chunks = split_bytes(data)
        assert len(chunks) == 1

    def test_data_larger_than_chunk_splits_correctly(self):
        data = b"x" * (self.CHUNK_SIZE + 1)
        chunks = split_bytes(data)
        assert len(chunks) == 2
        assert chunks[0] == b"x" * self.CHUNK_SIZE
        assert chunks[1] == b"x"

    def test_reassembly_matches_original(self):
        data = b"a" * (self.CHUNK_SIZE * 2 + 500)
        chunks = split_bytes(data)
        assert b"".join(chunks) == data

    def test_chunk_count(self):
        data = b"z" * (self.CHUNK_SIZE * 3)
        chunks = split_bytes(data)
        assert len(chunks) == 3
